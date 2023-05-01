import math
import os.path
import datasets
import json

def get_dataset(real_json, generated_json, subject, root_path):

    real_list = json.load(open(real_json, 'r'))[subject]
    generated_list = json.load(open(generated_json, 'r'))[subject]

    for i in range(len(real_list)):
        real_list[i]['img_path'] = os.path.join(root_path, real_list[i]['img_path'])

    for i in range(len(generated_list)):
        generated_list[i]['img_path'] = os.path.join(root_path, generated_list[i]['img_path'])

    repeat_time = int(math.ceil(len(generated_list) / len(real_list)))
    real_list = real_list * repeat_time
    real_list = real_list[: len(generated_list)]

    dataset_dict = {
        'real_image': list(item['img_path'] for item in real_list),
        'real_prompt': list(item['prompt'] for item in real_list),
        'generated_image': list(item['img_path'] for item in generated_list),
        'generated_prompt': list(item['prompt'] for item in generated_list)
    }

    features = datasets.Features({
        'real_image': datasets.Image(),
        'real_prompt': datasets.Value('string'),
        'generated_image': datasets.Image(),
        'generated_prompt': datasets.Value('string'),
    })

    dataset = datasets.Dataset.from_dict(dataset_dict, features, split="train")

    return dataset








