P_PER_NODE=$2
# # this should be same as ntasks-per-node/gpu_num

# ### init virtual environment if needed
source /gpfs/u/home/AICD/AICDzich/barn/miniconda3/etc/profile.d/conda.sh
conda activate swin

echo "P_PER_NODE"$P_PER_NODE
echo "SLURM_JOB_NUM_NODES"$SLURM_JOB_NUM_NODES
echo "SLURM_NODELIST"$SLURM_NODELIST

### the command to run
cd /gpfs/u/home/AICD/AICDzich/barn/code/multi_learning/mmsegmentation

NODE_RANK=${SLURM_PROCID}
ip2=dcs${SLURM_NODELIST:3:3}
NODE_LENGTH=${#SLURM_NODELIST}
if [[ ${NODE_LENGTH} == 6  ]]; then
    ip2=dcs${SLURM_NODELIST:3:3}
else
    ip2=dcs${SLURM_NODELIST:4:3}
fi

export MASTER_ADDR=${ip2}
export MASTER_PORT="8000"
export NODE_RANK=${NODE_RANK}

echo "MASTER_ADDR"${MASTER_ADDR}
echo "MASTER_PORT"${MASTER_PORT}
echo "NODE_RANK"${NODE_RANK}
echo "EXP"$3
EXP=$3

python -m torch.distributed.launch \
    --master_addr ${ip2} \
    --node_rank ${NODE_RANK} \
    --nproc_per_node ${P_PER_NODE}  \
    --nnodes $SLURM_JOB_NUM_NODES    \
    tools/train.py configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py --launcher pytorch \
    --auto-resume


# python -m torch.distributed.launch \
#     --master_addr ${ip2} \
#     --node_rank ${NODE_RANK} \
#     --nproc_per_node ${P_PER_NODE}  \
#     --nnodes $SLURM_JOB_NUM_NODES    \
#     train.py configs/cityscapes/simple_segnet/efficientvit-mobile/base2.yaml \
#     --rand_init kaiming_uniform@5 --last_gamma 0 \
#     --run_config "{base_lr:1.25e-4,n_epochs:250}" \
#     --data_provider "{resolution_config:{min:1024,max:2048},base_batch_size:2}" \
#     --data_provider.rrc_config.pad_mode random \
#     --data_provider.rrc_config.cat_max_ratio 0.6 \
#     --data_provider.extra_aug "[{name:color,mode:weak}]" \
#     --data_provider.cutmix_config "{alpha:0.5,stop_epoch:500,scale:null,same:true}" \
#     --ema_decay_list "[0.9998]" \
#     --net_config.backbone.path /gpfs/u/home/AICD/AICDzich/barn/code/mlcodebase/seg_and_cls/mobile_backbone/net.config \
#     --backbone_init_from /gpfs/u/home/AICD/AICDzich/barn/code/mlcodebase/seg_and_cls/mobile_backbone/init \
#     --alpha_init random@0.5 \
#     --run_config.criterion.main.max_to_keep 131072 \
#     --path /gpfs/u/home/AICD/AICDzich/scratch/work_dirs/Efficientvit/seg/cityscapes/simpleseg/mobile/base/tuning/2_alpha_init/random@0.5/run1

echo "FINISHED"

