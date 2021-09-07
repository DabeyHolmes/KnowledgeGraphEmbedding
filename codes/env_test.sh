

RANK_ID_START=1
RANK_SIZE=8

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
  python3.7.5 -c 'import torch;torch.npu.set_device('${RANK_ID}');x=torch.randn(2,3,224,224).npu();x+1'
  echo "device_"${RANK_ID}" check success"
done