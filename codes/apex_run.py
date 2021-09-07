#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn.parallel
import torch.distributed as dist

from torch.utils.data import DataLoader

from model import KGEModel
import model

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator

from apex import amp

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')



    # distributed training
    parser.add_argument('--backend', default='nccl', type=str,
                        help='Name of the backend to use')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='tcp://HOSTNAME:23455', type=str,
                        help='url used to set up distributed training')
    # parser.add_argument('--distributed', default='no', type=str)  # yes / no
    parser.add_argument('--distributed', action='store_true', help='Use distributed', default=False)
    parser.add_argument('--local_rank', default=0, type=int)

    # apex
    parser.add_argument('--apex', action='store_true', help='Use apex',default=False)
    parser.add_argument('--apex_level', default='O2', type=str)  # O1 / O2 优先使用 O2，若无法收敛则使用 O1
    parser.add_argument('--loss_scale', default=128.0, type=float)  # 优先使用 128.0，若无法收敛则使用 None

    # prof file
    # parser.add_argument('--prof', default='no', type=str)  # yes / no
    parser.add_argument('--prof', action='store_true', help='Use prof', default=False)
    
    return parser.parse_args(args)

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    
def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    if not args.distributed:
        entity_embedding = model.entity_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path, 'entity_embedding'),
            entity_embedding
        )
    
        relation_embedding = model.relation_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path, 'relation_embedding'),
            relation_embedding
        )

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w',

    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def myprint(str,args):
    if args.distributed:
        if args.do_train:
            log_file = os.path.join(args.save_path or args.init_checkpoint, 'print_train.log')
        else:
            log_file = os.path.join(args.save_path or args.init_checkpoint, 'print_test.log')
        with open(log_file,'a') as f:
            print(str,file=f)
            print(str)
def log_metrics(mode, step, metrics,args):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        myprint('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]),args)
        
        
def main(args):
    if args.distributed:
        # ===TODO===
        rank = args.local_rank
        dist_url = args.dist_url
        # torch.cuda.set_device(rank)
        # args.device = torch.device("cuda", args.local_rank)

        logging.info("Distributed Training (rank = {}), world_size = {}, backend = `{}', host-url = `{}'".format(
                     rank, args.world_size, args.backend, dist_url))
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port='23455')
        print(args.local_rank)
        dist.init_process_group(backend=args.backend, init_method=dist_init_method, world_size=args.world_size, rank=args.local_rank)
        logging.info("Distributed setting end!")
    # args.master_node = (not args.distributed) or (torch.distributed.is_initialized and torch.distributed.get_rank() == 0)
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Write logs to checkpoint and console
    set_logger(args)
    
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    # Read regions for Countries S* datasets
    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples
    
    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )
    
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()
    
    if args.do_train:
        # Set training dataloader iterator
        train_dataset_head = TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch')
        train_dataset_tail = TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch')
        nw = 0
        # nw = max(1, args.cpu_num//2)
        if args.distributed:
            train_sampler_head = torch.utils.data.distributed.DistributedSampler(train_dataset_head)
            train_sampler_tail = torch.utils.data.distributed.DistributedSampler(train_dataset_tail)
            train_dataloader_head = DataLoader(
                train_dataset_head,
                batch_size=args.batch_size,
                shuffle=(train_sampler_head is None),
                sampler=train_sampler_head,
                num_workers=nw,
                collate_fn=TrainDataset.collate_fn
            )
            train_dataloader_tail = DataLoader(
                train_dataset_head,
                batch_size=args.batch_size,
                shuffle=(train_sampler_tail is None),
                sampler=train_sampler_tail,
                num_workers=nw,
                collate_fn=TrainDataset.collate_fn
            )
        else:
            train_dataloader_head = DataLoader(
                train_dataset_head,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=nw,
                collate_fn=TrainDataset.collate_fn
            )

            train_dataloader_tail = DataLoader(
                train_dataset_tail,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=nw,
                collate_fn=TrainDataset.collate_fn
            )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        
        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2
        if args.apex:
            kge_model, optimizer = amp.initialize(kge_model, optimizer, opt_level=args.apex_level, loss_scale=args.loss_scale)
        if args.distributed:
            kge_model = torch.nn.parallel.DistributedDataParallel(kge_model, device_ids=[args.local_rank])

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        myprint('Loading checkpoint %s...' % args.init_checkpoint,args)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        myprint('Ramdomly Initializing %s Model...' % args.model,args)
        init_step = 0
    
    step = init_step
    print('train')
    logging.info('Start Training...')
    myprint('Start Training...',args)
    logging.info('init_step = %d' % init_step)
    myprint('init_step = %d' % init_step, args)
    logging.info('batch_size = %d' % args.batch_size)
    myprint('batch_size = %d' % args.batch_size, args)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    myprint('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling,args)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    myprint('hidden_dim = %d' % args.hidden_dim,args)
    logging.info('gamma = %f' % args.gamma)
    myprint('gamma = %f' % args.gamma,args)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    myprint('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling),args)
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
        myprint('adversarial_temperature = %f' % args.adversarial_temperature,args)
    
    # Set valid dataloader as it would be evaluated during training
    
    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)
        myprint('learning_rate = %d' % current_learning_rate,args)

        training_logs = []
        
        #Training Loop
        for step in range(init_step, args.max_steps):
            if args.distributed:
                train_sampler_head.set_epoch(step)
                train_sampler_tail.set_epoch(step)
            log = model.train_step(kge_model, optimizer, train_iterator, args)
            # print(step)
            # print(log)
            
            training_logs.append(log)
            
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                myprint('Change learning_rate to %f at step %d' % (current_learning_rate, step),args)
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()), 
                    lr=current_learning_rate
                )
                if args.apex:
                    kge_model, optimizer = amp.initialize(kge_model, optimizer, opt_level=args.apex_level, loss_scale=args.loss_scale)
                if args.distributed:
                    kge_model = torch.nn.parallel.DistributedDataParallel(kge_model, device_ids=[args.local_rank])
                warm_up_steps = warm_up_steps * 3
            
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)
                
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics,args)
                training_logs = []
                
            if args.do_valid and step % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                myprint('Evaluating on Valid Dataset...',args)
                metrics = model.test_step(kge_model, valid_triples, all_true_triples, args)
                log_metrics('Valid', step, metrics,args)
        
        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)
        
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        myprint('Evaluating on Valid Dataset...',args)
        metrics = model.test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics,args)
    
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        myprint('Evaluating on Test Dataset...',args)
        metrics = model.test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics,args)
    
    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        myprint('Evaluating on Training Dataset...',args)
        metrics = model.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics,args)
        
if __name__ == '__main__':
    main(parse_args())
