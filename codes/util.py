import os
def myprint(str,args):
    if args.distributed:
        if args.do_train:
            log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
        else:
            log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')
        with open(log_file,'a') as f:
            print(str,file=f)
            print(str)

def time_print(str,args):
    if args.distributed:
        if args.do_train:
            log_file = os.path.join(args.save_path or args.init_checkpoint, 'train_time.log')
        else:
            log_file = os.path.join(args.save_path or args.init_checkpoint, 'test_time.log')
        with open(log_file,'a') as f:
            print(str,file=f)
            # print(str)
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', start_count_index=10):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.start_count_index = start_count_index

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count == 0:
            self.N = n

        self.val = val
        self.count += n
        if self.count > (self.start_count_index * self.N):
            self.sum += val * n
            self.avg = self.sum / (self.count - self.start_count_index * self.N)

    def get_str(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)