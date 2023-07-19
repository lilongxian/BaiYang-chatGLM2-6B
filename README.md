# chatGLM2-6B BAIYANG探索版

优点：


1. 支持清华chatGLM-6B、alpaca指令微调训练数据格式。


2. 支持Lora量化训练，保留原始权重的能力。


3. 通过生成式的方法支持单任务和多任务知识挖掘类NLP任务。


4. 支持多轮对话、摘要、续写等知识生成类NLP任务。


5. 旋转位置词嵌入编码器支持2048区间弹性标准化，以取得在万级tokens上的更好效果（没有验证，理论上灵感来自于深度学习NLP的数据向量化过程中的数据标准化到[0.0，1.0]区间后训练较好这一个经验，值得关注的是伯克利Longchat做了验证。目前的GLM社区模型、LLAMA社区模型的基本单位tokens序列都是2048，所以可以以此为区间标准）。


6. 基于《大宋提刑官》证据论证+亚里斯多德三段论，探索出一种“基于证据理论的解释学习”机制，可有效提升模型解决数学、语言逻辑等复杂逻辑推理问题的能力。
具体的讲，我在原来alpaca指令数据的基础上增加了一个EXPLAIN，所以将response修改为：
   " EXPLAIN: " + explain部分 + " CONCLUSION: " + response部分
可参考数据集样例：data/explanation-based-learning-data


7. 上述“基于证据理论的解释学习”数据建模原理，可以拓展到高效复制chatgpt能力以提升GLM2A综合能力中。


本项目技术参考、引用了清华chatGLM-6B、chatGLM2-6B部分代码。若你使用本项目，请注明引用自清华chatGLM2-6B项目。本项目不保证商业化效果，仅供NLP学术研究。


# Update
1. 2023-07-04： 首次开源，仅次于清华官方微调训练模型开源时间。
2. 2023-07-11: 修复tokenization的上下句拼接方式，与llama、GLM2对齐.
3. 2023-07-11: 改进chatGLM2-6B官方的旋转位置词嵌入编码器。借鉴了longchat的可弹性压缩的位置标准化旋转编码器的设计方法，充分利用2048正弦波上的位置取得更好向量表示，以取得在万级tokens上的更好支持。
4. 2023-07-14: 改为Lora训练方式
5. 2023-07-19: 增加“基于证据理论的解释学习”数据建模机制，致力提升模型的复杂逻辑推理能力，并未高效从chatgpt等强大模型中复制能力奠定基础。

# 环境

cuda 11.7
pytorch 1.13.1/2.0
python 3.7/3.8
transformers 4.27.1--4.29.2

# Alpaca指令微调训练

alpaca数据转换命令：

     python data/fine-tuning-instraction-data/convert_alpaca2glm.py

证据理论解释学习的alpaca数据转换命令：

     python data/explanation-based-learning-data/convert_alpaca2glm_with_explain.py

将转换后的数据分配到 data/train.json


## GPU训练

1. 运行命令：

python finetune_norm_32k.py --do_train --train_file data/train.json  --history_column history  --prompt_column prompt --response_column response  --model_name_or_path D:/2023-LLM/PreTrained_LLM_Weights/chatGLM2-6B   --output_dir D:\glm_out\ --overwrite_output_dir --max_source_length 300 --max_target_length 200 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 10 --predict_with_generate --max_steps 500 --logging_steps 10 --save_steps 100 --learning_rate 1e-2 --quantization_bit 4 

--model_name_or_path 参数请修改为你的预训练模型所在目录


## 友情链接

https://github.com/THUDM/ChatGLM2-6B
https://github.com/DachengLi1/LongChat


## 参考

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

@misc{longchat2023,
    title = {How Long Can Open-Source LLMs Truly Promise on Context Length?},
    url = {https://lmsys.org/blog/2023-06-29-longchat},
    author = {Dacheng Li*, Rulin Shao*, Anze Xie, Ying Sheng, Lianmin Zheng, Joseph E. Gonzalez, Ion Stoica, Xuezhe Ma, and Hao Zhang},
    month = {June},
    year = {2023}
}


## 联系方式

whpxty5518@163.com / 17719085580（wx） Li·Long 


