#!/usr/bin/env python
# coding=utf-8
"""
# Copyright 2021 The HuggingFace Team. All rights reserved.
# Fine-tuning the library models for sequence to sequence.

声明：微调模型设计思路参考了清华官方 chatGLM-6B 版本：
https://github.com/THUDM/ChatGLM-6B/tree/main/ptuning

CreateTime: 2023-07-03
updateTime: 2023-07-11
Author: li-long·BaiYang

"""
import os
import torch
import datetime
import json
import numpy as np
from datasets import load_dataset
import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dataclasses import dataclass, field
from typing import Optional
from transformers import set_seed, \
    DataCollatorForSeq2Seq, HfArgumentParser, \
    Seq2SeqTrainingArguments
from glm2_core.trainer_seq2seq import Seq2SeqTrainer
from glm2_core.tokenization_chatglm import ChatGLMTokenizer
from glm2_core.modeling_chatglm import ChatGLMForConditionalGeneration
from glm2_core.configuration_chatglm import ChatGLMConfig
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None,
                                    metadata={"help": "The docment Path of pretrained model."})
    # 模型量化到的bit
    quantization_bit: Optional[int] = field(default=None)

    pre_seq_len: Optional[int] = field(
        default=128
    )
    # False
    prefix_projection: bool = field(
        default=False
    )


@dataclass
class DataTrainingArguments:
    prompt_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column prompt."})
    response_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column response."})
    history_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column history."})
    train_file: Optional[str] = field(
        default=None, metadata={"help": "training data file (a jsonlines or csv file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "a jsonlines or csv file."})
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "a jsonlines or csv file."})
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        })


@dataclass
class MyTrainingArguments(Seq2SeqTrainingArguments):
    trainable: Optional[str] = field(default="query_key_value,dense,dense_h_to_4h,dense_4h_to_h")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.)
    modules_to_save: Optional[str] = field(default="transformer.embedding.word_embeddings,transformer.output_layer,transformer.prefix_encoder.embedding")


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    model_and_tokenizer_config_dir = model_args.model_name_or_path

    """  加载数据集 """
    data_files = {}
    extension = None
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=training_args.output_dir,
        use_auth_token=None)

    """ 模型加载 """
    tokenizer = ChatGLMTokenizer.from_pretrained(model_and_tokenizer_config_dir)
    model_config_file = os.path.join(model_and_tokenizer_config_dir, "config.json")
    with open(model_config_file, "r", encoding="utf-8") as file:
        model_config = json.load(file)
    glm_config = ChatGLMConfig(**model_config)
    glm_config.pre_seq_len = model_args.pre_seq_len
    glm_config.prefix_projection = model_args.prefix_projection
    model = ChatGLMForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_and_tokenizer_config_dir,
        config=glm_config,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        load_in_8bit=False,
    )
    print(">>> Init new peft model...")  # 从原始的LLAMA模型开始用LoRA方法预训练
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=training_args.trainable.split(','),
        inference_mode=False,
        r=training_args.lora_rank,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        modules_to_save=training_args.modules_to_save.split(',') if training_args.modules_to_save else None
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if model_args.quantization_bit is not None:
        print("model_args.quantization_bit: ", model_args.quantization_bit)
        model = model.quantize(model_args.quantization_bit)
        model = model.half()
    if model_args.pre_seq_len is not None:
        model.transformer.prefix_encoder.float()

    """  数据向量化建模  """
    prefix = ""
    check = lambda x: 1 if x in raw_datasets else 0
    column_names = ""
    if training_args.do_train:
        assert check("train") == 1, "--do_train requires a train dataset!"
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        assert check("validation") == 1, "--do_eval requires a eval dataset!"
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        assert check("test") == 1, "--do_test requires a test dataset!"
        column_names = raw_datasets["test"].column_names

    prompt_column = data_args.prompt_column
    response_column = data_args.response_column
    history_column = data_args.history_column

    """ 核心：数据建模！！！"""

    def preprocess_function_train(examples):
        max_seq_length = data_args.max_source_length + data_args.max_target_length
        model_inputs = {"input_ids": [], "labels": []}
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query, answer = examples[prompt_column][i], examples[response_column][i]

                if history_column is None:
                    # ct数据建模策略，e.g: 摘要、单任务知识挖掘模型
                    prompt = query
                else:
                    # cqa数据建模策略，e.g: 对话、问答、多任务的知识挖掘模型
                    prompt = ""
                    history = examples[history_column][i]

                    if isinstance(history[0], str):
                        cqa_mode = "analysis"  # 多任务的文本分析，此时history也就是：[text]
                    else:
                        cqa_mode = "chat"  # 聊天对话，此时history的设计遵循清华chatGLM1-6B方法

                    if cqa_mode == "chat":
                        for i, (old_query, response) in enumerate(history):
                            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
                        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
                    elif cqa_mode == "analysis":
                        content = ""
                        for i, text in enumerate(history):
                            content += text
                        prompt_ = "给出以下信息：\n\n{}\n\n{}".format(content, query)
                        prompt = "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, prompt_)

                prompt = prefix + prompt
                a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
                b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

                # 则重于对话数据的上下文衔接
                if len(a_ids) > data_args.max_source_length - 1:
                    a_ids = a_ids[-data_args.max_source_length + 1:]
                if len(b_ids) > data_args.max_target_length - 1:
                    b_ids = b_ids[: data_args.max_target_length - 1]

                # 2023-07-11: GLM2这里已经不再使用[gMASK]，而是与LLAMA对齐:
                # [<bos>, a_tok_0, a_tok_1, ..., a_tok_m b_tok_0, b_tok_1,..., b_tok_n, <eos>]
                input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)
                context_length = len(a_ids) + 1
                labels = [-100] * context_length + input_ids[context_length:]
                assert len(input_ids) == len(labels), "error: len(input_ids) is not equal to len(labels)!"

                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len
                if data_args.ignore_pad_token_for_loss:
                    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

        return model_inputs

    def print_dataset_example(example):
        print("input_ids", example["input_ids"])
        print("inputs", tokenizer.tokenizer.decode(example["input_ids"]))
        print("label_ids", example["labels"])
        print("labels", tokenizer.tokenizer.decode(example["labels"]))

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function_train,
                batched=True,
                num_proc=None,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset")
        print_dataset_example(train_dataset[0])

    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False)

    """ 评价指标体系，rouge、bleu """

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        print("----decoded_preds----: \n", decoded_preds)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        print("----decoded_labels----: \n", decoded_labels)

        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": []
        }
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))
            result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

    print(""" Trainer """)
    training_args.generation_max_length = data_args.max_target_length
    training_args.no_cuda = False

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    """  --------- Training -------- """
    if training_args.do_train:
        print("*** Training ***")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        train_result = trainer.train()

        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics(split="train", metrics=metrics)
        trainer.save_metrics(split="train", metrics=metrics)  # Save metrics into a json file: `train_results.json`.
        trainer.save_state()
        print("---- 保存模型拓扑和权重完成 ----")



