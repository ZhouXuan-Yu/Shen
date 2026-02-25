# -*- coding: utf-8 -*-
import requests

# 测试各种搜索
tests = [
    ('ma', '拼音搜索'),
    ('妈妈', '中文搜索'),
    ('nǐ hǎo', '拼音完整搜索'),
]

print('测试搜索功能:')
print('-' * 50)

for query, desc in tests:
    r = requests.get(f'http://localhost:8000/api/v1/dictionary/search?query={query}')
    data = r.json()
    total = data['data']['total']
    print(f'{desc} "{query}": 找到 {total} 个结果')
    if total > 0:
        first = data['data']['results'][0]
        print(f'  第一个结果: chinese="{first["chinese"]}", pinyin="{first["pinyin"]}", score={first["score"]:.2f}')

