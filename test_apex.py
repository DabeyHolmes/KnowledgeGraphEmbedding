import os
# nl = [256]
# bl = [512]
# dl = [1000]
# gl = [24.0]
# al = [1.0]
# lrl = [0.0001]
# msl = [300000]
def tail(file, taillines=7, return_str=True):
    with open(file,'r') as f:
        lines = f.readlines()[-taillines:]
        print(lines)
    new_lines = []
    for line in lines:
        new_line = line.split(' ')[-1]
        new_lines.append(new_line)
    return "".join(new_lines) if return_str else new_lines
nl = [256]
bl = [512]
dl = [1000]
gl = [9.0]
al = [1.0]
lrl = [0.0001,0.0005,0.00005,0.00001]
msl = [300000]
# msl = [600]
apex_level = 'O1'
loss_scale = 'None' # None
date = '20210915_bs512_O1_None'
count = 0
world_size = 1
local_rank = 0
# sl = tail('models/RotatE_FB15k237_{}_{}/train.log'.format(date,count),return_str=False)
# print('ok')
# os.system( 'python -u codes/run.py --do_train --cuda --do_valid --do_test --data_path data/FB15k-237 --model RotatE -n 256 -b 512 -d 1000 -g 24.0 -a 1.0 -adv -lr 0.0001 --max_steps 300000 -save models/RotatE_FB15k_0 --test_batch_size 16 -de')
for n in nl:
    for b in bl:
        for d in dl:
            for g in gl:
                for a in al:
                    for lr in lrl:
                        for ms in msl:
                            print('hello')
                            # os.system('python -u codes/apex_run.py --do_train --cuda --do_valid --do_test --data_path data/FB15k-237 --model RotatE -n {} -b {} -d {} -g {} -a {} -adv -lr {} --max_steps {} -save models/RotatE_FB15k237_{}_{} --test_batch_size 16 -de --apex --apex_level {} --prof --distributed'.format(n,b,d,g,a,lr,ms,date,count,level))
                            os.system(
                                'python -u codes/apex_run.py --do_train --cuda --do_valid --do_test --data_path data/FB15k-237 --model RotatE -n {n} -b {b} -d {d} -g {g} -a {a} -adv -lr {lr} --max_steps {ms} -save models/RotatE_FB15k237_{date}_{count} --test_batch_size 16 -de --apex --apex_level {al} --loss_scale {ls} --prof --distributed --world_size {ws}'.format(
                                    n=n, b=b, d=d, g=g, a=a, lr=lr, ms=ms, date=date, count=count, al=apex_level, ls=loss_scale,ws=world_size))

                            sl = tail('models/RotatE_FB15k237_{}_{}/train.log'.format(date,count),return_str=False)
                            with open('models/RotatE_FB15k237_res_{}.csv'.format(date),'a') as f:
                                f.write('FB15k-237-{}-{}'.format(date,count))
                                f.write(',')
                                for s in sl:
                                    a = s.split(' ')[-1].split('\n')[0]
                                    f.write(a)
                                    f.write(',')
                                f.write('\r\n')
                            count += 1

