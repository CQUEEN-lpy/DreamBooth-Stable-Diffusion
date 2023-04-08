import json
path = './data/prompt_simple_generated.json'
with open(path, 'r') as f:
    dict = json.load(f)

for i in dict:
    print('------')
    for j in dict[i]:
        print(j)