from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch
import json
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
model_name = "/mnt1/zhangyuyang/code/LUNAR/quantized_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
quant_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
)

model.eval()

with open('/mnt1/zhangyuyang/code/LUNAR/data/forget.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]

prompt = '''You are an AI assistant whose task is to answer users' questions based on your own knowledge. If you don’t know the information about a question, you should apologize and refuse to answer.\n'''

for i in tqdm(data):
    input_text = prompt + "user: " + i['question'] + '\nAI: '
    inputs = tokenizer(input_text, return_tensors="pt").to(
        model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )

    full_response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    ai_response = full_response.split("AI: ")[-1]

    res = {
        'question': i['question'],
        'prediction': ai_response,
        'answer': i['answer']
    }

    with open('/mnt1/zhangyuyang/code/LUNAR/data/predict_forget_4b.jsonl', "a", encoding='utf-8') as fout:
        fout.write(json.dumps(res, ensure_ascii=False) + '\n')
        fout.flush()