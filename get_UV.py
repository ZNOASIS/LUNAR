from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from tqdm import tqdm

model_name = "/mnt1/open_source/models/tofu_llama2-7b"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

layer_accumulators = {idx: None for idx in range(15, 26)}
layer_averages = {}

prompt = '''You are an AI assistant whose task is to answer users' questions based on your own knowledge. If you don’t know the information about a question, you should apologize and refuse to answer.\n'''

def register_hook(layer_idx):
    def hook(module, input, output):
        current_output = torch.mean(output.detach()[0], dim=0)
        if layer_accumulators[layer_idx] is None:
            layer_accumulators[layer_idx] = torch.zeros_like(current_output)
        layer_accumulators[layer_idx] += current_output

        return output

    return hook


handles = []
for idx in range(15, 26):
    layer = model.model.layers[idx].mlp.down_proj
    handle = layer.register_forward_hook(register_hook(idx))
    handles.append(handle)

with open('/mnt1/zhangyuyang/code/LUNAR/data/fictious_clean.jsonl', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

for i in tqdm(data):
    input_text = prompt + "user: " + i['prompt'] + '\nAI: '
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)


data_size = len(data)
for layer_idx in layer_accumulators:
    if layer_accumulators[layer_idx] is not None:
        layer_averages[layer_idx] = layer_accumulators[layer_idx] / data_size

torch.save(layer_accumulators, "/mnt1/zhangyuyang/code/LUNAR/data/a_ref_f.pth")

layer_accumulators = {idx: None for idx in range(15, 26)}
layer_averages = {}

with open('/mnt1/zhangyuyang/code/LUNAR/data/forget.jsonl', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

for i in tqdm(data):
    input_text = prompt + "user: " + i['question'] + '\nAI: '
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)

data_size = len(data)
for layer_idx in layer_accumulators:
    if layer_accumulators[layer_idx] is not None:
        layer_averages[layer_idx] = layer_accumulators[layer_idx] / data_size

torch.save(layer_accumulators, "/mnt1/zhangyuyang/code/LUNAR/data/a_fg.pth")

