import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import swanlab

run = swanlab.init(
    project="LUNAR_top3",
    config={
        "learning_rate": 0.02,
        "epochs": 400,
    },
)

class ActivationDataset(Dataset):

    def __init__(self, activation_dict, tokenizer, max_length=128):
        self.prompts = list(activation_dict.keys())
        self.target_activations = list(activation_dict.values())
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'target_activation': self.target_activations[idx]
        }


def get_layer_activation(model, layer_idx):
    activation = None

    def hook(module, input, output):
        nonlocal activation
        activation = output[0]

    mlp_layer = model.model.layers[layer_idx].mlp
    hook_handle = mlp_layer.register_forward_hook(hook)

    return hook_handle, lambda: activation


def lunar_training(model, dataloader, layer_idx, lr=1e-2, epochs=10):

    for param in model.parameters():
        param.requires_grad = False

    target_layer = model.model.layers[layer_idx].mlp.down_proj
    target_layer.weight.requires_grad = True

    #optimizer = torch.optim.Adam([mlp_layer.down_proj.weight], lr=lr)
    optimizer = torch.optim.SGD(
        [target_layer.weight],
        lr=0.02,
        momentum=0.9,
        weight_decay=0,
        nesterov=True
    )

    hook_handle, get_activation = get_layer_activation(model, layer_idx)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(dataloader):
            optimizer.zero_grad()

            inputs = {
                'input_ids': batch['input_ids'].to(model.device),
                'attention_mask': batch['attention_mask'].to(model.device)
            }
            model(**inputs)

            current_activation = get_activation()
            target_activation = batch['target_activation'][0].to(model.device)

            loss = torch.nn.functional.mse_loss(
                current_activation.mean(dim=0),
                target_activation
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(dataloader):.4f}")
        swanlab.log({"loss": total_loss / len(dataloader)})

    hook_handle.remove()
    return model


MODEL_NAME = "/mnt1/open_source/models/tofu_llama2-7b"
LAYER_IDX = 21
BATCH_SIZE = 1
LR = 1e-2
EPOCHS = 400

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
weights = torch.load('/mnt1/zhangyuyang/code/LUNAR/data/snake-7.pt')
layer1 = model.model.layers[20].mlp.down_proj
layer1.load_state_dict(weights, assign=True)

activation_data = torch.load('/mnt1/zhangyuyang/code/LUNAR/data/train_data_top3.pth')[21]

dataset = ActivationDataset(activation_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

trained_model = lunar_training(
    model=model,
    dataloader=dataloader,
    layer_idx=LAYER_IDX,
    lr=LR,
    epochs=EPOCHS
)

torch.save(trained_model.model.layers[LAYER_IDX].mlp.down_proj.state_dict(), "/mnt1/zhangyuyang/code/LUNAR/data/lunar_down_proj_weights_layer21.pt")