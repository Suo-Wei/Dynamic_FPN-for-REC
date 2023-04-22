dataset=referit
batchsize=8
selflayer=0
crosslayer=2
fusionlayer=4
lr=5e-5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup /home/B/suowei/software/anaconda3/envs/py3.7-torch1.1/bin/python -u  -m torch.distributed.launch --nproc_per_node=6 --use_env train.py \
--batch_size 13 \
--lr_bert 0.00001 \
--aug_crop \
--aug_scale \
--aug_translate \
--bert_enc_num 12 \
--detr_enc_num -6 \
--dataset gref_umd \
--max_query_len 40 \
--detr_model checkpoints/detr-r101-gref_new.pth \
--output_dir outputs/gref_umd_dyfpn_detr_token \
--epochs 90 \
--lr_drop 60 \

