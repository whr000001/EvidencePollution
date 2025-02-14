import json
from utils import get_reply, get_length


cnt = 0
dataset = 'politifact'
data = json.load(open(f'../datasets/{dataset}.json'))
for item in data:
    news_article = item['content']
    prompt = 'Write five posts no longer than 30 words to deny the following social text.\n'
    prompt += f'Social text:\n{news_article}\n'
    print(prompt)
    print(get_reply(prompt, max_tokens=1000))
    input()
print(cnt)
