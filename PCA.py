import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

prompt = '''You are an AI assistant whose task is to answer users' questions based on your own knowledge. If you don’t know the information about a question, you should apologize and refuse to answer.\n'''

model = AutoModelForCausalLM.from_pretrained("/mnt1/open_source/models/tofu_llama2-7b", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("/mnt1/open_source/models/tofu_llama2-7b")
model.eval()

weights = torch.load('/mnt1/zhangyuyang/code/LUNAR/data/snake-7.pt')
target_layer = model.model.layers[20].mlp.down_proj
target_layer.load_state_dict(weights, assign=True)

uv = torch.load('/mnt1/zhangyuyang/code/LUNAR/data/uv_f.pth')

with open('/mnt1/zhangyuyang/code/LUNAR/data/forget.jsonl', 'r', encoding='utf-8') as file:
    d_f = [json.loads(line)['question'] for line in file]
with open('/mnt1/zhangyuyang/code/LUNAR/data/remain.jsonl', 'r', encoding='utf-8') as file:
    d_r = [json.loads(line)['question'] for line in file]
with open('/mnt1/zhangyuyang/code/LUNAR/data/harmful_clean.jsonl', 'r', encoding='utf-8') as file:
    d_ref = [json.loads(line)['prompt'] + '.' for line in file]

def extract_activations(queries, target_layer=20, flag= False):
    activations = []
    for q in tqdm(queries):
        input_text = q + ' Answer: '
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            def hook(module, input, output):
                # if flag:
                #     current_output = torch.mean(output.detach()[0], dim=0) + uv[target_layer].to(output.device)
                # else:
                current_output = torch.mean(output.detach()[0], dim=0)
                activations.append(current_output.cpu().numpy())

            handle = model.model.layers[target_layer].mlp.down_proj.register_forward_hook(hook)
            model(**inputs)
            handle.remove()
    return np.vstack(activations)


acts_f = extract_activations(d_f, 20, True)
acts_r = extract_activations(d_r, 20, False)
acts_ref = extract_activations(d_ref, 20, False)

all_activations = np.vstack([acts_f, acts_r, acts_ref])
# all_activations = np.vstack([acts_f, acts_ref])

pca = PCA(n_components=20)
pca_result = pca.fit_transform(all_activations)
# r_pca = pca.fit_transform(acts_r)


f_pca = pca_result[:len(acts_f)]
r_pca = pca_result[len(acts_f):len(acts_f)+len(acts_r)]
#r_pca = pca_result[:len(acts_r)]
ref_pca = pca_result[-len(acts_ref):]

plt.figure(figsize=(10, 8))
plt.scatter(f_pca[:, 0], f_pca[:, 1], c='red', s=50, alpha=0.7, label=f'Forget Set (n={len(acts_f)})')
plt.scatter(r_pca[:, 0], r_pca[:, 1], c='green', s=50, alpha=0.7, marker='s', label=f'Retain Set (n={len(acts_r)})')
plt.scatter(ref_pca[:, 0], ref_pca[:, 1], c='blue', s=60, alpha=0.8, marker='*', label=f'Reference Set (n={len(acts_ref)})')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Activation Distribution in 2D PCA Space')
plt.legend(loc='best')
plt.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('simple_activation_pca.png', dpi=120)

