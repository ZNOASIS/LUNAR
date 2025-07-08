from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

model_name = "/data3/zhangyuyang/tofu_llama2-7b"
fout_path = "/data3/zhangyuyang/LUNAR/harmful_clean.jsonl"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # 自动将模型分配到可用GPU
    torch_dtype=torch.float16,  # 使用半精度减少显存占用
)


with open('/data3/zhangyuyang/LUNAR/harmful.json', 'r') as file:
    data = json.load(file)

for i in tqdm(data):
    prompt = i['instruction'] + '.'

    inputs = tokenizer.encode(
        prompt,
        return_tensors='pt'
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=1000)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_length = len(prompt)
    generated_text_without_input = generated_text[prompt_length:].strip()

    if len(generated_text_without_input) >= 10:
        res = {
            'prompt': prompt,
            'answer': generated_text_without_input
        }
        with open(fout_path, "a", encoding='utf-8') as fout:
            fout.write(json.dumps(res, ensure_ascii=False) + '\n')
            fout.flush()
        print(generated_text_without_input)
