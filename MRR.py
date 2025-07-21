from transformers import AutoTokenizer
import numpy as np
import json

tokenizer = AutoTokenizer.from_pretrained("/mnt1/open_source/models/tofu_llama2-7b")

def calculate_mrr(generated_answers, ground_truths, tokenizer):
    mrr_scores = []

    for generated_answer, ground_truth in zip(generated_answers, ground_truths):
        generated_tokens = tokenizer.tokenize(generated_answer)
        ground_truth_tokens = tokenizer.tokenize(ground_truth)
        ranks = []
        for token in ground_truth_tokens:
            if token in generated_tokens:
                rank = generated_tokens.index(token) + 1
                ranks.append(rank)

        if ranks:
            reciprocal_ranks = [1.0 / rank for rank in ranks]
            mrr_scores.append(np.mean(reciprocal_ranks))
        else:
            mrr_scores.append(0)
    return np.mean(mrr_scores)

with open('/mnt1/zhangyuyang/code/LUNAR/data/base_forget.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]

ground_truths = [i['answer'] for i in data]
generated_answers = [i['prediction'] for i in data]

mrr = calculate_mrr(generated_answers, ground_truths, tokenizer)
print(mrr)

with open('/mnt1/zhangyuyang/code/LUNAR/data/base_remain.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]

ground_truths = [i['answer'] for i in data]
generated_answers = [i['prediction'] for i in data]

mrr = calculate_mrr(generated_answers, ground_truths, tokenizer)
print(mrr)