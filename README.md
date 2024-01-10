## 基本设置
1. 运行环境：Python 3.10、Pytorch 2.1.2
2. 硬件环境：CPU Intel(R) Xeon(R) Gold 5118
            GPU TITAN V
3. 基座模型：chinese-alpaca-2-1.3b
4. 微调数据集：alpaca_gpt4_zh
5. 微调方法：LORA 学习率：2e-5 训练轮数：2.0
## 训练命令
```bash
CUDA_VISIBLE_DEVICES=4 python src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path model/chinese-alpaca-2-1.3b \
    --finetuning_type lora \
    --template default \
    --dataset_dir data \
    --dataset alpaca_gpt4_zh \
    --cutoff_len 1024 \
    --learning_rate 2e-05 \
    --num_train_epochs 2.0 \
    --max_samples 10000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --neftune_noise_alpha 0 \
    --lora_rank 8 \
    --lora_dropout 0.1 \
    --lora_target q_proj,v_proj \
    --output_dir saves/ChineseLLaMA2-1.3B/lora/train_2024-01-10-00-17-52 \
    --fp16 True \
    --plot_loss True
```
## 过程截图
1. 数据准备
<img width="696" alt="15bb937e78b248da4651d68816c75f9" src="https://github.com/HenryChang666/LLaMA-Factory/assets/156045275/bc6cdbf2-8404-4b4e-b6a3-16df4bedb956">

2. 模型训练
<img width="939" alt="77f11b525d405cc5e7e3cce82ec99c1" src="https://github.com/HenryChang666/LLaMA-Factory/assets/156045275/af88094c-e9f9-4b48-a4d6-41fd5bcfd239">

3. 训练完成
<img width="974" alt="image" src="https://github.com/HenryChang666/LLaMA-Factory/assets/156045275/e867f15f-5c4d-406d-b499-4b087e1716dc">
