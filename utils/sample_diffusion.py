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
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        required=False,
        help="the pretrained checkpoint link",
    )

    parser.add_argument(
        "--num_cls",
        type=int,
        default=10,
        required=False,
        help="How many images are generated per class",
    )

    parser.add_argument(
        "--generated_json",
        type=str,
        default="../data/generated.json",
        required=False,
        help="the path where prompt(generated) json is stored",
    )

    parser.add_argument(
        "--real_json",
        type=str,
        default="../data/real.json",
        required=False,
        help="the path where prompt(real) json is stored",
    )

    parser.add_argument(
        '--subject',
        type=str,
        default='pink_sunglasses',
        help='The subject id',
    )

    parser.add_argument(
        '--cls',
        type=str,
        default='glasses',
        help='The class id',
    )

    parser.add_argument(
        '--save_path',
        type=str,
        default='../data/img/generated',
        help='The generated image path',
    )

    parser.add_argument(
        '--subject_path',
        type=str,
        default='../data/img/subject',
        help='The subject image path',
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    config = parse_args()

    try:
        prompt_dict = json.load(open(config.generated_json, 'r'))
    except:
        print("generated_json does not exist, try to create one ")
        tmp_dict = {}
        with open(config.generated_json, 'w') as f:
            json.dump(tmp_dict, f)

    try:
        real_dict = json.load(open(config.real_json, 'r'))
    except:
        print("real_json does not exist, try to create one ")
        tmp_dict = {}
        with open(config.real_json, 'w') as f:
            json.dump(tmp_dict, f)

    os.makedirs(config.save_path, exist_ok=True)
    subject_path = os.path.join(config.save_path, config.subject)
    os.makedirs(subject_path, exist_ok=True)

    # create the dict for generated images
    prompt_list = []
    for i in range(config.num_cls):
        tmp_path = os.path.join(subject_path, str(i).zfill(5)+'.jpg')
        tmp = {"img_path": "/".join(os.path.normpath(tmp_path).split(os.sep)[-3:]),
               "prompt": 'a [V] ' + config.cls}
        prompt_list.append(tmp)
    prompt_dict[config.subject] = prompt_list
    with open(config.generated_json, 'w') as f:
        json.dump(prompt_dict, f)

    # create the corresponded dict for real images
    tmp_list = []

    for file in os.listdir(os.path.join(config.subject_path, config.subject)):
        if '.jpg' in file:
            path = os.path.join(config.subject_path, config.subject, file)
            item = {
                'img_path': "/".join(os.path.normpath(path).split(os.sep)[-3:]),
                'prompt': f'a [V] {config.cls}'
            }

            tmp_list.append(item)

    real_dict[config.subject] = tmp_list

    with open(config.real_json, 'w') as f:
        json.dump(real_dict, f)

    pipe = StableDiffusionPipeline.from_pretrained(config.model_id, torch_dtype=torch.float32)
    pipe = pipe.to("cuda")

    progress_bar = tqdm(total=len(prompt_list))
    progress_bar.set_description(f"Generating Images for subject: {config.subject}")

    for item in prompt_list:
        prompt = item['prompt']
        prompt = prompt.replace('[V]', '', 3)
        prompt = re.sub(r'\s+', ' ', prompt)

        if os.path.exists(item['img_path']):
            print(f'skipping {item["img_path"]}')
            continue

        image = pipe(prompt).images[0]
        tmp_path = os.path.join(config.img_path, item['img_path'])
        image.save(tmp_path)
        progress_bar.update(1)