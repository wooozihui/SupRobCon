# SupRobCon

This is the code of our new work " A simple supervised robust constrastive learning framework " (temporary)

This code is based on pytorch and well supports the pytorch ddp training.

## Preparation
For now, this code only supports the CIFAR-10 dataset, and there is a script "setup.sh" to 
automatic create the dir "data" and download the dataset into it.
Just run as:
```
git clone https:/github.com/wooozihui/SupRobCon
cd SupRobCon
./setup.sh
```

## Train SupRobCon
After the preparation, you can adjust the run.sh to train your own SupRobCon model.
Note there are two crucial parameters, they are:

```
CUDA_VISIBLE_DEVICES: this parameter makes your GPUs visble into runtime. If you only have one card, please set CUDA_VISIBLE_DEVICES=0.
nproc_per_node: Set this parameter as your GPU numbers.
```

After setting these two parameters, you should be able to run this code as:

```
./run.sh
```

Unlike the previous CL works such as SimCLR and MoCo or SupCon, this code using an
online linear evaluatoin, which means the FC layer is trained on-the-fly while 
training the backbone, so there is no need to run a linear evaluatoin.

## Set your tensorboard
You may want to monitor the training process using the tensorboard, this code is also supported.

### Firstly, you should run the tensorboard
```
tensorboard --logdir your_log_path(e.g., ./logs) --bind_all --port your_port(e.g., 23333)
```

### Then set the parameter tb_path in run.sh
```
--tb_path = your_log_path
```

Then the robustness and accuracy, and the training loss will be recorded while training.





