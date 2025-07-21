from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
from tqdm import tqdm

model_name = "/mnt1/zhangyuyang/models_finetune/pistol_sample1/llama2-7b-chat"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("/mnt1/open_source/models/Llama2-7b-chat")

layer_accumulators = {idx: None for idx in range(23, 32)}
layer_averages = {}

def register_hook(layer_idx):
    def hook(module, input, output):
        current_output = torch.mean(output.detach()[0], dim=0)
        if layer_accumulators[layer_idx] is None:
            layer_accumulators[layer_idx] = torch.zeros_like(current_output)
        layer_accumulators[layer_idx] += current_output

        return output

    return hook


handles = []
for idx in range(23, 32):
    layer = model.model.layers[idx].mlp.down_proj
    handle = layer.register_forward_hook(register_hook(idx))
    handles.append(handle)

with open('/mnt1/zhangyuyang/code/LUNAR/data/harmful_clean.jsonl', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

for i in tqdm(data):
    input_text = i['prompt'] + '. Answer: '
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)


data_size = len(data)
for layer_idx in layer_accumulators:
    if layer_accumulators[layer_idx] is not None:
        layer_averages[layer_idx] = layer_accumulators[layer_idx] / data_size

torch.save(layer_accumulators, "/mnt1/zhangyuyang/code/LUNAR/data/a_ref_pistol.pth")

layer_accumulators = {idx: None for idx in range(23, 32)}
layer_averages = {}

with open('/mnt1/zhangyuyang/code/LUNAR/data/forget_pistol.jsonl', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

for i in tqdm(data):
    input_text = i['question'] + ' Answer: '
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)

data_size = len(data)
for layer_idx in layer_accumulators:
    if layer_accumulators[layer_idx] is not None:
        layer_averages[layer_idx] = layer_accumulators[layer_idx] / data_size

torch.save(layer_accumulators, "/mnt1/zhangyuyang/code/LUNAR/data/a_fg_pistol.pth")

