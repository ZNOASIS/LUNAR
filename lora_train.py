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
    project="LUNAR",
    config={
        "learning_rate": 0.01,
        "epochs": 300,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    },
)


# 新增：LoRA适配器实现
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank, alpha, dropout=0.0):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha

        for param in self.original.parameters():
            param.requires_grad = False

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.lora_A, mean=0, std=0.02)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        original_output = self.original(x)
        lora_output = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * (self.alpha / self.rank)
        return original_output + lora_output

    def merge_weights(self):
        merged_weight = self.original.weight + (self.lora_B @ self.lora_A) * (self.alpha / self.rank)
        return merged_weight


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

def lunar_training(model, dataloader, layer_idx, lr=1e-2, epochs=10, lora_rank=8, lora_alpha=16, lora_dropout=0.05):
    for param in model.parameters():
        param.requires_grad = False

    target_layer = model.model.layers[layer_idx].mlp.down_proj

    lora_layer = LoRALayer(
        original_layer=target_layer,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout
    )
    model.model.layers[layer_idx].mlp.down_proj = lora_layer

    optimizer = torch.optim.SGD(
        [
            lora_layer.lora_A,
            lora_layer.lora_B
        ],
        lr=lr,
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

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")
        swanlab.log({"loss": avg_loss})

    hook_handle.remove()
    lora_weights = {
        "lora_A": lora_layer.lora_A.detach().cpu(),
        "lora_B": lora_layer.lora_B.detach().cpu(),
        "alpha": lora_alpha,
        "rank": lora_rank
    }
    return model, lora_weights

MODEL_NAME = "/mnt1/open_source/models/tofu_llama2-7b"
LAYER_IDX = 20
BATCH_SIZE = 1
LR = 1e-2
EPOCHS = 300
LORA_RANK = run.config.lora_rank
LORA_ALPHA = run.config.lora_alpha
LORA_DROPOUT = run.config.lora_dropout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

activation_data = torch.load('/mnt1/zhangyuyang/code/LUNAR/data/train_data.pth')

dataset = ActivationDataset(activation_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

trained_model, lora_weights = lunar_training(
    model=model,
    dataloader=dataloader,
    layer_idx=LAYER_IDX,
    lr=LR,
    epochs=EPOCHS,
    lora_rank=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT
)

torch.save(lora_weights, "/mnt1/zhangyuyang/code/LUNAR/data/lora.pt")