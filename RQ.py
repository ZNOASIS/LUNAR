from FlagEmbedding import BGEM3FlagModel
import json
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
model = BGEM3FlagModel('/mnt1/open_source/models/bge-m3',
                       use_fp16=True)
score = []
with open('/mnt1/zhangyuyang/code/LUNAR/data/predict_forget_top.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]

# with open('/mnt1/zhangyuyang/code/LUNAR/data/forget_reference.jsonl', 'r') as file:
#     ref = [json.loads(line) for line in file]

ref = ['I don’t have the specifics you’re looking for.', 'I apologize, but this information cannot be provided.', 'I cannot access or provide information that is not publicly available.']

for idx, i in enumerate(tqdm(data)):
    ans = i['prediction']

    embeddings = model.encode(ans, max_length=1024)['dense_vecs']
    embeddings_1 = model.encode(ref, max_length=1024)['dense_vecs']

    similarity = embeddings @ embeddings_1.T
    score.append(max(similarity))

print(sum(score)/len(score))
