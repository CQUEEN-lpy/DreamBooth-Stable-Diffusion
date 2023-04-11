import math
import os.path
import datasets
from torch.utils import data
import json
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


def get_dataset(real_json, generated_json, subject, root_path, transform=None):

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

def collate_fn(item):


if __name__ == '__main__':
    preprocess = transforms.Compose(
        [
            transforms.Resize((512, 512),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )

    dataset = get_dataset('../data/prompt_simple_real.json', '../data/prompt_simple_generated.json', 'dog6', '../data/img', transform = preprocess)
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    indexs = np.random.randint(0, len(dataset), 4)
    sample_data = dataset[indexs]
    for i in range(len(indexs)):
        axs[i].imshow(sample_data['real_image'][i])
        axs[i].set_axis_off()
        axs[i].text(0.5, -0.2, sample_data['real_prompt'][i], fontsize=10, ha='center', transform=axs[i].transAxes)

    plt.show()








