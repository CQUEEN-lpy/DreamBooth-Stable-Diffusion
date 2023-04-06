import json

with open('./data/class.info', 'r') as f:
    dict = {}
    for line in f:
        line = line[:-1] if '\n' in line else line
        subject, cls = line.split(',')
        dict[subject] = cls

with open('./data/class.json', 'w') as f:
    json.dump(dict, f)
    #testss