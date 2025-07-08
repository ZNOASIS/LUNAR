import json

with open('/data3/zhangyuyang/TOFU/full.json', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]


first_20_data = data[:20]
remaining_data = data[20:]


with open('/data3/zhangyuyang/LUNAR/forget.jsonl', 'w', encoding='utf-8') as file:
    for item in first_20_data:
        file.write(json.dumps(item, ensure_ascii=False) + '\n')


with open('/data3/zhangyuyang/LUNAR/remain.jsonl', 'w', encoding='utf-8') as file:
    for item in remaining_data:
        file.write(json.dumps(item, ensure_ascii=False) + '\n')

print("数据分割完成！")
