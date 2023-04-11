import math
import os.path

from torch.utils import data
import json
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

class DreamBooth_Dataset(data.Dataset):
    def __init__(self, real_json, generated_json, subject, root_path, transform=None):

        self.real_list = json.load(open(real_json, 'r'))[subject]
        self.generated_list = json.load(open(generated_json, 'r'))[subject]
        self.transform = transform
        self.root_path = root_path

        self.real_images = []
        self.real_prompts = []
        for item in self.real_list:
            real_img = Image.open(os.path.join(self.root_path, item['img_path'])).convert("RGB")
            self.real_images.append(real_img)
            self.real_prompts.append(item['prompt'])

        self.generated_images = []
        self.generated_prompts = []
        for item in self.real_list:
            generated_img = Image.open(os.path.join(self.root_path, item['img_path'])).convert("RGB")
            self.generated_images.append(generated_img)
            self.generated_prompts.append(item['prompt'])

        repeat_time = int(math.ceil(len(self.generated_prompts) / len(self.real_images)))
        self.real_images = self.real_images * repeat_time
        self.real_images = self.real_images[: len(self.generated_images)]
        self.real_prompts = self.real_prompts * repeat_time
        self.real_prompts = self.real_prompts[: len(self.generated_images)]

    def __len__(self):
        return len(self.generated_list)

    def __getitem__(self, index):

        real_imgs = self.real_images[index]
        generated_imgs = self.generated_images[index]

        if self.transform:
            real_imgs = self.transform(real_imgs)
            generated_imgs = self.transform(generated_imgs)

        real_prompts = self.real_prompts[index]
        generated_prompts = self.generated_prompts[index]

        return {
            'real_img': real_imgs,
            'real_prompt': real_prompts,
            'generated_img': generated_imgs,
            'generated_prompt': generated_prompts,
        }


if __name__ == '__main__':
    preprocess = transforms.Compose(
        [
            transforms.Resize((512, 512),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )

    dataset = DreamBooth_Dataset('../data/prompt_simple_real.json', '../data/prompt_simple_generated.json', 'dog6', transform=preprocess, root_path='../data/img')

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    indexs = np.random.randint(0, len(dataset), 4)
    sample_data = dataset[indexs]
    for i in range(len(indexs)):
        axs[i].imshow(sample_data['real_img'][i][0])
        axs[i].set_axis_off()
        axs[i].text(0.5, -0.2, sample_data['real_prompt'][i][1], fontsize=10, ha='center', transform=axs[i].transAxes)





