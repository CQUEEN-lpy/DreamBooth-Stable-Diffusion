import diffusers.utils.logging
from diffusers.utils import logging
logging.disable_progress_bar()
from diffusers import StableDiffusionPipeline
import torch
import argparse, os, json
from tqdm.auto import tqdm
import re

def parse_args():
    parser = argparse.ArgumentParser(description="generating images using the frozen pretrained diffusion model")

    parser.add_argument(
        "--num_per_epoch",
        type=int,
        default=1,
        required=False,
        help="How many images are simple_generated in one epoch",
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
        default=1,
        required=False,
        help="How many images are simple_generated per class",
    )

    parser.add_argument(
        "--prompt_json",
        type=str,
        default="../data/prompt_simple_generated.json",
        required=False,
        help="the path where prompt json is stored",
    )

    parser.add_argument(
        "--img_path",
        type=str,
        default='../data/img',
        required=False,
        help="the path where class json is stored",
    )

    parser.add_argument(
        '--subjects',
        nargs='+',
        type=str,
        default=['red_cartoon'],
        help='The subject id',
    )

    parser.add_argument(
        '--save_path',
        type=str,
        default='../data/img/simple_generated',
        help='The subject id',
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    config = parse_args()
    diffusers.utils.logging.disable_progress_bar()


    prompt_dict = json.load(open(config.prompt_json, 'r'))

    pipe = StableDiffusionPipeline.from_pretrained(config.model_id, torch_dtype=torch.float32)
    pipe = pipe.to("cuda")
    os.makedirs(config.save_path, exist_ok=True)

    for subject in config.subjects:
        prompt_list = prompt_dict[subject]

        dir_path = os.path.join(config.save_path, subject)
        os.makedirs(dir_path, exist_ok=True)
        progress_bar = tqdm(total=len(prompt_list))
        progress_bar.set_description(f"Generating Images for subject: {subject}")


        for item in prompt_list:
            prompt = item['prompt']
            prompt = prompt.replace('[V]', '', 3)
            prompt = re.sub(r'\s+', ' ', prompt)

            if os.path.exists(os.path.join(config.img_path, item['img_path'])):
                print(f'skipping {item["img_path"]}')
                continue

            image = pipe(prompt).images[0]
            tmp_path = os.path.join(config.img_path, item['img_path'])
            image.save(tmp_path)
            progress_bar.update(1)




