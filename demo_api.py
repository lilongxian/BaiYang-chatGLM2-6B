"""
声明：copy from chatGLM2-6B
2023-06-29
"""

from fastapi import FastAPI, Request
import uvicorn, json, datetime
import torch
import os
from glm2_core.tokenization_chatglm import ChatGLMTokenizer as AutoTokenizer
from glm2_core.modeling_chatglm import ChatGLMForConditionalGeneration as AutoModel, ChatGLMConfig as AutoConfig


DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


@app.post("/chatglm2")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    response, history = model.chat(tokenizer,
                                   prompt,
                                   history=history,
                                   max_length=max_length if max_length else 2048,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer


if __name__ == '__main__':
    model_and_tokenizer_config_dir = "D:/2023-LLM/PreTrained_LLM_Weights/chatGLM2-6B/"
    tokenizer = AutoTokenizer.from_pretrained(model_and_tokenizer_config_dir)
    model_config_file = os.path.join(model_and_tokenizer_config_dir, "config.json")
    with open(model_config_file, "r", encoding="utf-8") as file:
        model_config = json.load(file)
    glm_config = AutoConfig(**model_config)
    model = AutoModel.from_pretrained(model_and_tokenizer_config_dir, config=glm_config)
    model = model.quantize(4)
    model.to("cuda:0")

    # 多显卡支持，使用下面三行代替上面两行，将num_gpus改为你实际的显卡数量
    # model_path = "THUDM/chatglm2-6b"
    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # from glm2_core.utils import load_model_on_gpus
    # model = load_model_on_gpus(model_path, num_gpus=2)
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
