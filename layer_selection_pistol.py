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
uv = torch.load('/mnt1/zhangyuyang/code/LUNAR/data/uv_f.pth')

hooks = []
prompt = '''You are an AI assistant whose task is to answer users' questions based on your own knowledge. If you don’t know the information about a question, you should apologize and refuse to answer.\n'''

flag = {20:False, 21:False, 23:False}
ans = []
ref = []

layer1 = model.model.layers[20].mlp.down_proj
layer2 = model.model.layers[21].mlp.down_proj
layer3 = model.model.layers[23].mlp.down_proj

def hook1(module, input, output):
    vector = uv[20].to(output.device)
    if not flag[20]:
        flag[20] = True
        output[0] = output[0] + vector
    return output
def hook2(module, input, output):
    vector = uv[21].to(output.device)
    if not flag[21]:
        flag[21] = True
        output[0] = output[0] + vector
    return output
def hook3(module, input, output):
    vector = uv[23].to(output.device)
    if not flag[23]:
        flag[23] = True
        output[0] = output[0] + vector
    return output
hooks.append(layer1.register_forward_hook(hook1))
hooks.append(layer2.register_forward_hook(hook2))
#hooks.append(layer3.register_forward_hook(hook3))

with open('/mnt1/zhangyuyang/code/LUNAR/data/forget_reference.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]

for i in tqdm(data):

    ref.append(i['reference'])
    input_text = prompt + "user: " + i['question'] + '\nAI: '
    inputs = tokenizer(input_text, return_tensors="pt").to(
        model.device)

    flag[20] = False
    flag[21] = False
    flag[23] = False

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )

    full_response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    ai_response = full_response.split("AI: ")[-1]
    print(ai_response)
    ans.append(ai_response)

res = {
    'answer': ans,
    'reference': ref
}

with open('/mnt1/zhangyuyang/code/LUNAR/data/layer_selection_top3.jsonl', "a", encoding='utf-8') as fout:
    fout.write(json.dumps(res, ensure_ascii=False) + '\n')
    fout.flush()


