CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=4 --master_port 12333 test.py  \
--model_path suprobcon_temp_0.5_supcon_adv \
--load_best 1 \
--total_num 1000 \
--test_bs 250 \
--attack_type pgd_logits_scale \
--alpha 1
