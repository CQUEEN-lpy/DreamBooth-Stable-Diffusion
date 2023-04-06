from dataclasses import dataclass
from diffusers import StableDiffusionPipeline
import torch
import argparse, os, json
from tqdm.auto import tqdm
from diffusers.utils import logging

#logging.disable_progress_bar()

def parse_args():
    parser = argparse.ArgumentParser(description="generating images using the frozen pretrained diffusion model")

    parser.add_argument(
        "--num_per_epoch",
        type=int,
        default=1,
        required=False,
        help="How many images are generated in one epoch",
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        required=False,
        help="the pretrained checkpoint link",
    )

    parser.add_argument(
        "--num_cls",
        type=int,
        default=1000,
        required=False,
        help="How many images are generated per class",
    )

    parser.add_argument(
        "--prompt_json",
        type=str,
        default="../data/prompt_simple.json",
        required=False,
        help="the path where prompt json is stored",
    )

    parser.add_argument(
        "--class_json",
        type=str,
        default='../data/class.json',
        required=False,
        help="the path where class json is stored",
    )

    parser.add_argument(
        '--subjects',
        nargs='+',
        type=str,
        default=['fancy_boot', 'dog6', 'cat2', 'red_cartoon', 'teapot'],
        help='The subject id',
    )

    parser.add_argument(
        '--save_path',
        type=str,
        default='../data/img/generated',
        help='The subject id',
    )

    args = parser.parse_args()
    return args





"""pipe = StableDiffusionPipeline.from_pretrained(config.model_id, torch_dtype=torch.float32)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

image.save("astronaut_rides_horse.png")"""

if __name__ == '__main__':
    config = parse_args()
    class_dict = json.load(open(config.class_json, 'r'))
    prompt_dict = json.load(open(config.prompt_json, 'r'))
    assert len(class_dict) * len(prompt_dict) > 0

    pipe = StableDiffusionPipeline.from_pretrained(config.model_id, torch_dtype=torch.float32)
    pipe = pipe.to("cuda")

    for subject in config.subjects:
        cls = class_dict[subject]
        prompt_list = prompt_dict[cls]
        count = config.num_cls
        counter = 0
        dir_path = os.path.join(config.save_path, subject)
        progress_bar = tqdm(total=config.num_cls)
        progress_bar.set_description(f"Generating Images for subject: {subject}")

        while count > 0:

            for prompt in prompt_list:
                num_images_per_prompt = config.num_per_epoch if count - config.num_per_epoch > 0 else count
                count += -config.num_per_epoch

                images = pipe(prompt, num_images_per_prompt=num_images_per_prompt).images
                for image in images:
                    path = os.path.join(dir_path, f'{str(counter).zfill(5)}.jpg')
                    image.save(path)
                    counter += 1
                    progress_bar.update(1)



