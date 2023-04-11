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

    def __len__(self):
        return len(self.generated_list)

    def __getitem__(self, index):
        real_index = len(self.generated_list)

        real_img = Image.open(os.path.join(self.root_path, self.real_list[real_index]['img_path'])).convert("RGB")
        generated_img = Image.open(os.path.join(self.root_path, self.generated_list[index]['img_path'])).convert("RGB")

        if self.transform:
            real_img = self.transform(real_img)
            generated_img = self.transform(generated_img)

        real_prompt = self.real_list[real_index]['prompt']
        generated_prompt = self.generated_list[index]['prompt']

        return {
            'real': (real_img, real_prompt),
            'generated': (generated_img, generated_prompt)
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
        axs[i].imshow(sample_data['real'][i][0])
        axs[i].set_axis_off()
        axs[i].text(0.5, -0.2, sample_data['real'][i][1], fontsize=10, ha='center', transform=axs[i].transAxes)





