from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import torch
from tqdm import tqdm

class SkipLayer:
    def __init__(self, model, layer_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.hook = None

    def __enter__(self):
        def skip_forward(module, input, output):
            return input[0]

        self.hook = model.model.layers[self.layer_idx].mlp.register_forward_hook(skip_forward)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook:
            self.hook.remove()


os.environ["CUDA_VISIBLE_DEVICES"] = "5"
model_name = "/mnt1/open_source/models/tofu_llama2-7b"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open('/mnt1/zhangyuyang/code/LUNAR/data/forget.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]

with SkipLayer(model, 21):
    weights = torch.load('/mnt1/zhangyuyang/code/LUNAR/data/rabbit-8.pt')
    target_layer = model.model.layers[20].mlp.down_proj
    target_layer.load_state_dict(weights, assign=True)

    weights2 = torch.load('/mnt1/zhangyuyang/code/LUNAR/data/lunar_down_proj_weights.pt')
    target_layer = model.model.layers[21].mlp.down_proj
    target_layer.load_state_dict(weights2, assign=True)

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

        with open('/mnt1/zhangyuyang/code/LUNAR/data/predict_forget_skip.jsonl', "a", encoding='utf-8') as fout:
            fout.write(json.dumps(res, ensure_ascii=False) + '\n')
            fout.flush()