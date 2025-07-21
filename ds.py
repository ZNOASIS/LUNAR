from rouge_score import rouge_scorer
import json
import math

r_f = 0
r_r = 0
scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

with open('/mnt1/zhangyuyang/code/LUNAR/data/predict_forget_4b.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]

for i in data:
    pre = i['prediction']
    ref = i['answer']
    r_f += scorer.score(ref, pre)['rouge1'][1]

r_f /= len(data)
print(r_f)

with open('/mnt1/zhangyuyang/code/LUNAR/data/predict_forget_paraphrase.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]

for i in data:
    pre = i['prediction']
    ref = i['answer']
    s = scorer.score(ref, pre)['rouge1'][1]
    r_r += s

r_r /= len(data)
print(r_r)

ds = 100 * math.sqrt(r_f*r_f+(1-r_r)*(1-r_r))

print(ds)



