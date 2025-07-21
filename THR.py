import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

model_name = "/mnt1/open_source/models/tofu_llama2-7b"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open('/mnt1/zhangyuyang/code/LUNAR/data/forget_paraphrase.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]

weights1 = torch.load('/mnt1/zhangyuyang/code/LUNAR/data/rabbit-8.pt')
target_layer = model.model.layers[20].mlp.down_proj
target_layer.load_state_dict(weights1, assign=True)

prompt = '''You are an AI assistant whose task is to answer users' questions based on your own knowledge. If you don’t know the information about a question, you should apologize and refuse to answer.\n'''

for i in tqdm(data):
    input_text = prompt + "user: " + i['reference'] + '\nAI: '
    inputs = tokenizer(input_text, return_tensors="pt").to(
        model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    print(logits)
    break
