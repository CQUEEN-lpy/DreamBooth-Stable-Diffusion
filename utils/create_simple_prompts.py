import os
import json



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
            prompt_dict[i] = []

            for file in os.listdir(os.path.join(subject_path, i)):
                if '.jpg' in file:
                    path = os.path.join(subject_path, i, file)
                    item = {
                        'img_path': "/".join(os.path.normpath(path).split(os.sep)[-3:]),
                        'prompt': f'a [V] {cls}'
                    }

                    prompt_dict[i].append(item)

        json.dump(prompt_dict, f)

def create_simple_prompt_generated(num=1000):
    with open('../data/class.json', 'r') as f:
        class_dict = json.load(f)

    with open('../data/prompt_simple_generated.json', 'w') as f:
        prompt_dict = {}
        generated_path = '../data/img/simple_generated'
        subject_path = '../data/img/subject'

        for i in os.listdir(subject_path):
            try:
                cls = class_dict[i]
            except:
                continue
            prompt_dict[i] = []

            for file_id in range(num):
                file_name = f'{str(file_id).zfill(5)}.jpg'
                path = os.path.join(generated_path, i, file_name)

                item = {
                    'img_path': "/".join(os.path.normpath(path).split(os.sep)[-3:]),
                    'prompt': f'a [V] {cls}'
                }

                prompt_dict[i].append(item)

        json.dump(prompt_dict, f)

def create_eval(path, subject):
    try:
        dict = json.load(open(path, 'r'))
    except:
        dict = {}
    dict[subject] = [
        'A woman wearing [V] boots, strolling through a market.',
        'A pair of [V] boots floating in zero gravity.',
        'A [V] boot carefully placed on a train platform.',
        'A bike rider with [V] boots leisurely pedaling.',
        'a red [V] boot',
        'batman wearing [V] boots'
    ]

    with open(path, 'w') as f:
        json.dump(dict, f)

if __name__ == '__main__':
    create_eval('../data/eval.json', 'fancy_boot')