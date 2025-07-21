import json
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    api_key= "sk-6771d938d9584f4a86d6d97324c67e1b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

prompt = '''You are an expert in text rewriting. Your task is to rephrase a question while ensuring that its semantics and subject remain unchanged.'''

with open('/mnt1/zhangyuyang/code/LUNAR/data/forget.jsonl', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

for i in tqdm(data):
    query = i['question']
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
        extra_body={"enable_thinking": False}
    )
    res = {
        'question': i['question'],
        'reference': completion.choices[0].message.content,
        'answer': i['answer']
    }
    with open('/mnt1/zhangyuyang/code/LUNAR/data/forget_paraphrase.jsonl', "a", encoding='utf-8') as fout:
        fout.write(json.dumps(res, ensure_ascii=False) + '\n')
        fout.flush()