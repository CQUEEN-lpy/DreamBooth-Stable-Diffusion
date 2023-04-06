import json
with open('./data/prompt_simple.json', 'r') as f:
    dict = json.load(f)
    for i in dict:
        print(i)
        for j in dict[i]:
            print(j)
        print('-----------')