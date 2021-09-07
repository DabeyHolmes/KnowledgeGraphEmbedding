import os
# nl = [256]
# bl = [512]
# dl = [1000]
# gl = [24.0]
# al = [1.0]
# lrl = [0.0001]
# msl = [300000]
def tail(file, taillines=5, return_str=True, avg_line_length=None):
    """avg_line_length:每行字符平均数,
    return_str:返回类型，默认为字符串，False为列表。
    offset:每次循环相对文件末尾指针偏移数"""
    with open(file, errors='ignore') as f:
        if not avg_line_length:
            f.seek(0, 2)
            f.seek(f.tell() - 3000)
            avg_line_length = int(3000 / len(f.readlines())) + 10
        f.seek(0, 2)
        end_pointer = f.tell()
        offset = taillines * avg_line_length
        if offset > end_pointer:
            f.seek(0, 0)
            lines = f.readlines()[-taillines:]
            return "".join(lines) if return_str else lines
        offset_init = offset
        i = 1
        while len(f.readlines()) < taillines:
            location = f.tell() - offset
            f.seek(location)
            i += 1
            offset = i * offset_init
            if f.tell() - offset < 0:
                f.seek(0, 0)
                break
        else:
            f.seek(end_pointer - offset)
        lines = f.readlines()
        if len(lines) >= taillines:
            lines = lines[-taillines:]

        return "".join(lines) if return_str else lines
nl = [32,64,128,256,512]
bl = [512]
dl = [1000]
gl = [24.0]
al = [1.0]
lrl = [0.0001]
msl = [300000]
date = '20210907'
count = 0
# os.system( 'python -u codes/run.py --do_train --cuda --do_valid --do_test --data_path data/FB15k-237 --model RotatE -n 256 -b 512 -d 1000 -g 24.0 -a 1.0 -adv -lr 0.0001 --max_steps 300000 -save models/RotatE_FB15k_0 --test_batch_size 16 -de')
for n in nl:
    for b in bl:
        for d in dl:
            for g in gl:
                for a in al:
                    for lr in lrl:
                        for ms in msl:
                            os.system(
                                'python -u codes/run.py --do_train --cuda --do_valid --do_test --data_path data/FB15k-237 --model RotatE -n {} -b {} -d {} -g {} -a {} -adv -lr {} --max_steps {} -save models/RotatE_FB15k237_{}_{} --test_batch_size 16 -de --apex --prof'.format(n,b,d,g,a,lr,ms,date,count))

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

