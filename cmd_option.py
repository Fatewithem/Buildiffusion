# tensorboard --logdir / --port 6006 --host 0.0.0.0
# ssh -L 6006:localhost:6006 root@172.31.233.186

# CUDA_VISIBLE_DEVICES=0,1 python main_co3d.py

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1


