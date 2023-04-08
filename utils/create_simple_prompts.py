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


def create_simple_prompt_real():
    with open('../data/class.json', 'r') as f:
        class_dict = json.load(f)

    with open('../data/prompt_simple_real.json', 'w') as f:

        subject_path = '../data/img/subject'
        prompt_dict = {}

        for i in os.listdir(subject_path):
            try:
                cls = class_dict[i]
            except:
                continue
            prompt_dict[cls] = []

            for file in os.listdir(os.path.join(subject_path, i)):
                if '.jpg' in file:
                    path = os.path.join(subject_path, i, file)
                    item = {
                        'img_path': "/".join(os.path.normpath(path).split(os.sep)[-3:]),
                        'prompt': f'a [V] {cls}'
                    }

                    prompt_dict[cls].append(item)

        json.dump(prompt_dict, f)


if __name__ == '__main__':
    create_simple_prompt_real()