CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 12333 test.py  \
--model_path suprobcon_temp_1_mutistep  \
--load_best 1 \
--total_num 1000 \
--test_bs 250 \
--attack_type AA
