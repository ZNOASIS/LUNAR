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
uv = torch.load('/mnt1/zhangyuyang/code/LUNAR/data/uv_f.pth')

hooks = []
prompt = '''You are an AI assistant whose task is to answer users' questions based on your own knowledge. If you don’t know the information about a question, you should apologize and refuse to answer.\n'''

for idx in range(15, 26):
    ans = []
    ref = []
    for hook in hooks:
        hook.remove()
    layer = model.model.layers[idx].mlp.down_proj

    def hook(module, input, output, layer_idx=idx):
        vector = uv[layer_idx].to(output.device)
        if not hasattr(hook, 'is_called'):
            setattr(hook, 'is_called', True)
            output[0] = output[0] + vector
        return output


    hook_handle = layer.register_forward_hook(hook)
    hooks.append(hook_handle)

    with open('/mnt1/zhangyuyang/code/LUNAR/data/forget_reference.jsonl', 'r') as file:
        data = [json.loads(line) for line in file]

    for i in tqdm(data):
        ref.append(i['reference'])
        input_text = prompt + "user: " + i['question'] + '\nAI: '
        inputs = tokenizer(input_text, return_tensors="pt").to(
            model.device)

        if hasattr(hook, 'is_called'):
            delattr(hook, 'is_called')

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id
            )

        full_response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        ai_response = full_response.split("AI: ")[-1]
        ans.append(ai_response)

    res = {
        'answer': ans,
        'reference': ref
    }
    with open('/mnt1/zhangyuyang/code/LUNAR/data/layer_selection.jsonl', "a", encoding='utf-8') as fout:
        fout.write(json.dumps(res, ensure_ascii=False) + '\n')
        fout.flush()


