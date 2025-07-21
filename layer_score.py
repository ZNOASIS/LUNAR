from FlagEmbedding import BGEM3FlagModel
import json
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
model = BGEM3FlagModel('/mnt1/open_source/models/bge-m3',
                       use_fp16=True)
score = []
with open('/mnt1/zhangyuyang/code/LUNAR/data/layer_selection.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]

for i in tqdm(data):
    ans = i['answer']
    ref1 = i['reference']
    ref2 = ["I apologize, but I cannot fulfill your request. It is unethical and illegal.", "I apologize, but I cannot fulfill your request, because it is unethical and potentially illegal."]

    embeddings = model.encode(ans, max_length=1024)['dense_vecs']
    embeddings_1 = model.encode(ref1, max_length=1024)['dense_vecs']
    embeddings_2 = model.encode(ref2, max_length=1024)['dense_vecs']
    similarity_1 = embeddings @ embeddings_1.T
    similarity_2 = embeddings @ embeddings_2.T

    s = 0

    for idx, j in enumerate(similarity_1):
        s += j[idx]
    # for idx, j in enumerate(similarity_2):
    #     s -= max(j[0], j[1])
        # [[0.6265, 0.3477], [0.3499, 0.678 ]]

    score.append(s)

print(score)
