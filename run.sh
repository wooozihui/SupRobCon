CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 --master_port 12333 main_src.py  \
--savepath simsiam_test \
--in_d 512 \
--out_d 2048 \
--train_bs 1024 \
--scheduler cosineanneal \
--training_epoch 100 \
--warm_epoch 10 \
--init_lr 0.1 \
--scale_factor 8 \
--temperature 0.5 \
--save_last_best 1

:<<!
self.mlp_head = nn.Sequential(
            nn.Linear(feature_d, feature_d),
            nn.BatchNorm1d(feature_d),
            nn.ReLU(inplace=True),
            nn.Linear(feature_d, mlp_d),
            #nn.ReLU(),
        )
!