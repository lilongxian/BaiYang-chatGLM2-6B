# LL-BaiYang

本项目对清华2023-06-25开源的chatGLM2-6B模型进行了模型网络建设、数据建模等全部技术的反向研究（关键地方做了注释，并修复了tokenization脚本中的bug）。并尝试复现其指令微调训练模型。

优点：

1. 支持清华chatGLM-6B、alpaca指令微调训练数据格式。

2. 支持GPU量化训练、CPU全参数训练。

3. 支持单任务和多任务的知识挖掘。

4. 支持多轮对话。

5. 支持2048区间弹性标准化的旋转位置词嵌入编码器，以取得在万级tokens上的更好效果。

本项目技术参考、引用了清华chatGLM-6B、chatGLM2-6B部分代码。若你使用本项目，请注名引用清华知识产权。本项目开源，不保证商业化效果，仅供学术研究使用。


# Update
1. 2023-07-04： 首次开源，仅次于清华官方微调训练模型开源时间。
2. 2023-07-11: 修复tokenization的上下句拼接方式，与llama、GLM2对齐.
3. 2023-07-11: 改进chatGLM2-6B官方的旋转位置词嵌入编码器。借鉴了longchat的可弹性压缩的位置标准化旋转编码器的设计方法，充分利用2048正弦波上的位置取得更好向量表示，以取得在万级tokens上的更好支持。


# 环境

cuda 11.7
pytorch 1.13.1/2.0
python 3.7/3.8
transformers 4.27.1--4.29.2

# Alpaca指令微调训练

alpaca数据转换命令：python convert_alpaca2glm.py

将转换后的数据分配到 data/train.json

## CPU训练

请确保你的空闲内存足够。当seq_length为500时，将消耗不低于58G内存！

1. 注释掉finetuning.py中的os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

2. 运行命令：

python finetune_norm_32k.py --do_train --train_file data/train.json  --history_column history  --prompt_column prompt --response_column response  --model_name_or_path D:/2023-LLM/PreTrained_LLM_Weights/chatGLM2-6B   --output_dir D:\glm_out\ --overwrite_output_dir --max_source_length 300 --max_target_length 200 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 1 --predict_with_generate --max_steps 500 --logging_steps 1 --save_steps 100 --learning_rate 1e-2 

--model_name_or_path 参数请修改为你的预训练模型所在目录

## GPU训练

1. 运行命令：

python finetune_norm_32k.py --do_train --train_file data/train.json  --history_column history  --prompt_column prompt --response_column response  --model_name_or_path D:/2023-LLM/PreTrained_LLM_Weights/chatGLM2-6B   --output_dir D:\glm_out\ --overwrite_output_dir --max_source_length 300 --max_target_length 200 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 10 --predict_with_generate --max_steps 500 --logging_steps 10 --save_steps 100 --learning_rate 1e-2 --quantization_bit 4 

--model_name_or_path 参数请修改为你的预训练模型所在目录


# 友情链接

https://github.com/THUDM/ChatGLM2-6B

@inproceedings{du2022glm,
  title={GLM: General Language Model Pretraining with Autoregressive Blank Infilling},
  author={Du, Zhengxiao and Qian, Yujie and Liu, Xiao and Ding, Ming and Qiu, Jiezhong and Yang, Zhilin and Tang, Jie},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={320--335},
  year={2022}
}

@article{zeng2022glm,
  title={Glm-130b: An open bilingual pre-trained model},
  author={Zeng, Aohan and Liu, Xiao and Du, Zhengxiao and Wang, Zihan and Lai, Hanyu and Ding, Ming and Yang, Zhuoyi and Xu, Yifan and Zheng, Wendi and Xia, Xiao and others},
  journal={arXiv preprint arXiv:2210.02414},
  year={2022}
}

https://github.com/DachengLi1/LongChat

@misc{longchat2023,
    title = {How Long Can Open-Source LLMs Truly Promise on Context Length?},
    url = {https://lmsys.org/blog/2023-06-29-longchat},
    author = {Dacheng Li*, Rulin Shao*, Anze Xie, Ying Sheng, Lianmin Zheng, Joseph E. Gonzalez, Ion Stoica, Xuezhe Ma, and Hao Zhang},
    month = {June},
    year = {2023}
}

# 联系方式

whpxty5518@163.com / 17719085580（wx） Li·Long 


