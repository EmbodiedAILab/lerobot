# Install

```bash
conda create -n lerobot-npu python=3.10
conda activate lerobot-npu

# 会有报错，可忽略
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 torch_npu==2.3.1.post4
pip install -r requirements-npu.txt
```

# Test policy

使用仿真进行测试，由于昇腾卡一般没有桌面，故在这里xvfb提供虚拟桌面并使用cpu进行相关图片的渲染，实际在机器人使用的时候并不需要此前缀

## Test Diffusion policy

pretrained open source weights: https://hf-mirror.com/lerobot/diffusion_pusht

```bash
xvfb-run -a -s "-screen 0 1600x900x30" python lerobot/scripts/eval.py --policy.path=lerobot/diffusion_pusht --env.type=pusht --eval.batch_size=10 --eval.n_episodes=10 --policy.device=npu
```

## Test ACT

pretrained open source weights: https://hf-mirror.com/lerobot/act_aloha_sim_transfer_cube_human

```bash
xvfb-run -a -s "-screen 0 1600x900x30" python lerobot/scripts/eval.py --policy.path=lerobot/act_aloha_sim_transfer_cube_human --env.type=aloha --env.task=AlohaTransferCube-v0 --eval.batch_size=10 --eval.n_episodes=10 --policy.device=npu
```

## Test PI0

```bash
# pi0没有开源的训练好的面向任务的chkpt，需要先自己简单训一下
# 这里为了测试，把保存频率和steps都配置得非常小，实际使用默认值或更大
python lerobot/scripts/train.py --output_dir=outputs/train/pi0_aloha_transfer --policy.type=pi0 --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --wandb.enable=false --steps=200 --save_freq=200

xvfb-run -a -s "-screen 0 1600x900x30" python lerobot/scripts/eval.py --policy.path=outputs/train/pi0_aloha_transfer/checkpoints/000200/pretrained_model --env.type=aloha --env.task=AlohaTransferCube-v0 --eval.batch_size=1 --eval.n_episodes=1 --policy.device=npu
```

### 性能打点特性

请见`lerobot/common/policies/pi0/modeling_pi0.py`文件的`select_action`函数，首次调用会触发一次模型推理（第281-第300行），一次推理输出50个action，放入队列`_action_queue`中，消费完成后才会触发下一次推理。同时，首次推理涉及到一些初始化，耗时较长，建议进行打点时进行额外记录。

### 关于PI0的补充说明

PI0是北美Physical Intelligence公司的开源模型，其[官方实现](https://github.com/Physical-Intelligence/openpi)基于Jax框架，而Jax框架目前不支持昇腾。本代码仓为huggingface lerobot社区参照其Jax代码实现的torch版本，对昇腾友好。

 Install

```bash
conda create -n lerobot-npu python=3.10
conda activate lerobot-npu

# 会有报错，可忽略
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 torch_npu==2.3.1.post4
pip install -r requirements-npu.txt
```

# Test policy

使用仿真进行测试，由于昇腾卡一般没有桌面，故在这里xvfb提供虚拟桌面并使用cpu进行相关图片的渲染，实际在机器人使用的时候并不需要此前缀

## Test Diffusion policy

pretrained open source weights: https://hf-mirror.com/lerobot/diffusion_pusht

```bash
xvfb-run -a -s "-screen 0 1600x900x30" python lerobot/scripts/eval.py --policy.path=lerobot/diffusion_pusht --env.type=pusht --eval.batch_size=10 --eval.n_episodes=10 --policy.device=npu
```

## Test ACT

pretrained open source weights: https://hf-mirror.com/lerobot/act_aloha_sim_transfer_cube_human

```bash
xvfb-run -a -s "-screen 0 1600x900x30" python lerobot/scripts/eval.py --policy.path=lerobot/act_aloha_sim_transfer_cube_human --env.type=aloha --env.task=AlohaTransferCube-v0 --eval.batch_size=10 --eval.n_episodes=10 --policy.device=npu
```

## Test PI0

```bash
# pi0没有开源的训练好的面向任务的chkpt，需要先自己简单训一下
# 这里为了测试，把保存频率和steps都配置得非常小，实际使用默认值或更大
python lerobot/scripts/train.py --output_dir=outputs/train/pi0_aloha_transfer --policy.type=pi0 --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --wandb.enable=false --steps=200 --save_freq=200

xvfb-run -a -s "-screen 0 1600x900x30" python lerobot/scripts/eval.py --policy.path=outputs/train/pi0_aloha_transfer/checkpoints/000200/pretrained_model --env.type=aloha --env.task=AlohaTransferCube-v0 --eval.batch_size=1 --eval.n_episodes=1 --policy.device=npu
```

### 性能打点特性

请见`lerobot/common/policies/pi0/modeling_pi0.py`文件的`select_action`函数，首次调用会触发一次模型推理（第281-第300行），一次推理输出50个action，放入队列`_action_queue`中，消费完成后才会触发下一次推理。同时，首次推理涉及到一些初始化，耗时较长，建议进行打点时进行额外记录。

### 关于PI0的补充说明

PI0是北美Physical Intelligence公司的开源模型，其[官方实现](https://github.com/Physical-Intelligence/openpi)基于Jax框架，而Jax框架目前不支持昇腾。本代码仓为huggingface lerobot社区参照其Jax代码实现的torch版本，对昇腾友好。

值得注意的是，由于本工作较新，其torch版本在实际机器人上运行的效果从多方反馈及实测，暂时仍逊色于官方的Jax版本，社区仍在不断优化中。但从推理速度测试的角度，已经可以提供一个具有说服力的参考。