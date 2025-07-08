from transformers import LlamaForCausalLM, LlamaTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch

model_name = "/data3/zhangyuyang/tofu_llama2-7b"
model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
model.eval()
tokenizer = LlamaTokenizer.from_pretrained(model_name)

outputs = {}

def register_hook(layer_idx):
    def hook(module, input, output):
        outputs[f"layer_{layer_idx}"] = output.detach()
    return hook

handles = []
for idx in range(10, 23):
    layer = model.model.layers[idx].mlp.down_proj
    handle = layer.register_forward_hook(register_hook(idx))
    handles.append(handle)


input_text = "Who is this celebrated LGBTQ+ author from Santiago, Chile known for their true crime genre work?"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
with torch.no_grad():
    model(**inputs)

for handle in handles:
    handle.remove()

print(outputs["layer_11"].shape)