import os
import json

with open('../data/class.json', 'r') as f:
    class_dict = json.load(f)

with open('../data/prompt_simple.json', 'w') as f:

    subject_path = '../data/img/subject'
    prompt_dict = {}

    for i in os.listdir(subject_path):
        try:
            cls = class_dict[i]
        except:
            continue
        prompt_dict[cls] = []

        num_prompt = len(os.listdir(os.path.join(subject_path, i)))
        for i in range(num_prompt):
            prompt_dict[cls].append(f'a [V] {cls}')

    json.dump(prompt_dict, f)