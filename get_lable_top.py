from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from tqdm import tqdm

model_name = "/mnt1/open_source/models/tofu_llama2-7b"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

data = {20: {}, 21: {}, 23: {}}

uv = torch.load('/mnt1/zhangyuyang/code/LUNAR/data/uv_f.pth')
prompt = '''You are an AI assistant whose task is to answer users' questions based on your own knowledge. If you don’t know the information about a question, you should apologize and refuse to answer.\n'''

def register_hook(prompt, flag, idx):
    def hook(module, input, output):
        current_output = torch.mean(output.detach()[0], dim=0)
        if flag:
            data[idx][prompt] = current_output
        else:
            data[idx][prompt] = current_output + uv[idx].to(output.device)

        return output
    return hook


with open('/mnt1/zhangyuyang/code/LUNAR/data/remain.jsonl', 'r') as file:
    remain = [json.loads(line) for line in file]
with open('/mnt1/zhangyuyang/code/LUNAR/data/forget.jsonl', 'r') as file:
    forget = [json.loads(line) for line in file]


layer1 = model.model.layers[20].mlp.down_proj
layer2 = model.model.layers[21].mlp.down_proj
layer3 = model.model.layers[23].mlp.down_proj

for i in tqdm(remain):

    handle1 = layer1.register_forward_hook(register_hook(i['question'], True, 20))
    handle2 = layer2.register_forward_hook(register_hook(i['question'], True, 21))
    handle3 = layer3.register_forward_hook(register_hook(i['question'], True, 23))
    input_text = prompt + "user: " + i['question'] + '\nAI: '
    inputs = tokenizer(input_text, return_tensors="pt").to(
        model.device)

    with torch.no_grad():
        model(**inputs)

    handle1.remove()
    handle2.remove()
    handle3.remove()

for i in tqdm(forget):
    handle1 = layer1.register_forward_hook(register_hook(i['question'], False, 20))
    handle2 = layer2.register_forward_hook(register_hook(i['question'], False, 21))
    handle3 = layer3.register_forward_hook(register_hook(i['question'], False, 23))
    input_text = prompt + "user: " + i['question'] + '\nAI: '
    inputs = tokenizer(input_text, return_tensors="pt").to(
        model.device)

    with torch.no_grad():
        model(**inputs)

    handle1.remove()
    handle2.remove()
    handle3.remove()

torch.save(data, "/mnt1/zhangyuyang/code/LUNAR/data/train_data_top3.pth")

