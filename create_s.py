import json
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    api_key= "sk-6771d938d9584f4a86d6d97324c67e1b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

prompt = '''You are a text analysis expert. Your task is to refuse to answer questions provided by users, and you need to explain that the refusal behavior is due to a lack of knowledge. For example, if a user asks: Can you tell us about the type of books that Jaime Vasquez writes? An example answer is: I apologize, but I cannot provide information on the types of books written by Jaime Vasquez as I do not have access to his personal information or literary works.'''''

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
        'reference': completion.choices[0].message.content
    }
    with open('/mnt1/zhangyuyang/code/LUNAR/data/forget_reference.jsonl', "a", encoding='utf-8') as fout:
        fout.write(json.dumps(res, ensure_ascii=False) + '\n')
        fout.flush()