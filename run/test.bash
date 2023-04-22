dataset=referit
batchsize=8
selflayer=0
crosslayer=2
fusionlayer=4
lr=5e-5

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 nohup /home/suowei/anaconda3/envs/py3.7-torch1.1/bin/python3.7 -u  -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py \
--batch_size 32 \
--lr_bert 0.00001 \
--aug_crop \
--aug_scale \
--aug_translate \
--bert_enc_num 12 \
--detr_enc_num -6 \
--dataset gref_umd \
--max_query_len 40 \
--eval_set test \
--eval_model ./outputs/gref_umd_dyfpn_detr/best_checkpoint.pth \
--output_dir ./outputs/gref_umd_dyfpn_detr \
--epochs 90 \
--lr_drop 60 \

