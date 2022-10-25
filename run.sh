export HF_HOME=/data/wangyuanchun/NLP_course/from_HF
cmd="CUDA_VISIBLE_DEVICES=4,5 \
python -m \
torch.distributed.launch \
--nproc_per_node=2 \
--nnodes=1 \
--node_rank=0 \
--master_addr='127.0.0.1' --master_port='29500' \
main.py \
2>&1 | tee log.txt"

echo $cmd
eval $cmd