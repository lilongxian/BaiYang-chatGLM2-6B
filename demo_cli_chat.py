"""
声明：copy from chatGLM2-6B
2023-06-29
"""

import os, json
import platform
from glm2_core.tokenization_chatglm import ChatGLMTokenizer as AutoTokenizer
from glm2_core.modeling_chatglm import ChatGLMForConditionalGeneration as AutoModel, ChatGLMConfig as AutoConfig


model_and_tokenizer_config_dir = "D:/2023-LLM/PreTrained_LLM_Weights/chatGLM2-6B/"
tokenizer = AutoTokenizer.from_pretrained(model_and_tokenizer_config_dir)
model_config_file = os.path.join(model_and_tokenizer_config_dir, "config.json")
with open(model_config_file, "r", encoding="utf-8") as file:
    model_config = json.load(file)
glm_config = AutoConfig(**model_config)
glm_config.pre_seq_len = None  # 禁用前置编码器，你也可以设置一个合理的int数值来启用她
glm_config.prefix_projection = False
model = AutoModel.from_pretrained(model_and_tokenizer_config_dir, config=glm_config)
model = model.quantize(4).half()
model.to("cuda:0")
print(">>> 微调模型权重参数加载完成！")

# 多显卡支持，使用下面两行代替上面的模型加载，将num_gpus改为你实际的显卡数量
# from utils import load_model_on_gpus
# model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM2-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    past_key_values, history = None, []
    global stop_stream
    print("欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print("欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        print("\nChatGLM：", end="")
        current_length = 0
        for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
            if stop_stream:
                stop_stream = False
                break
            else:
                print(response[current_length:], end="", flush=True)
                current_length = len(response)
        print("")


if __name__ == "__main__":
    main()
