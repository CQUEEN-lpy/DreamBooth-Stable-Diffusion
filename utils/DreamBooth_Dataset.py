import os.path

from torch.utils import data
import json
from PIL import Image
from torchvision import transforms

class DreamBooth_Dataset(data.Dataset):
    def __init__(self, real_json, generated_json, subject, root_path, transform=None):

        self.real_list = json.load(open(real_json, 'r'))[subject]
        self.generated_list = json.load(open(generated_json, 'r'))[subject]
        self.transform = transform
        self.root_path = root_path

    def __len__(self):
        print(len(self.generated_list))

    def __getitem__(self, index):
        real_index = index % len(self.generated_list)

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

    dataset = DreamBooth_Dataset('../data/prompt_simple_real.json', '../data/prompt_simple_generated.json', 'dog', transform=preprocess)





