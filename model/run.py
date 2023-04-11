import argparse, os, random, string, json
from tqdm.auto import tqdm
import numpy as np
from utils.DreamBooth_Dataset import get_dataset
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import functools
from torchvision import transforms
import re
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="generating images using the frozen pretrained diffusion model")

    parser.add_argument(
        "--subject",
        type=str,
        default='dog6',
        required=False,
        help="The subject we want to finetune on",
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        required=False,
        help="the pretrained checkpoint link",
    )

    parser.add_argument(
        "--identifier_len",
        type=int,
        default=3,
        required=False,
        help="the length of the random identifier",
    )

    parser.add_argument(
        "--real_path",
        type=str,
        default='../data/prompt_simple_real.json',
        required=False,
        help="the json path for real json",
    )

    parser.add_argument(
        "--generated_path",
        type=str,
        default='../data/prompt_simple_generated.json',
        required=False,
        help="the json path for generated json",
    )

    parser.add_argument(
        "--img_path",
        type=str,
        default='../data/img',
        required=False,
        help="the path where class json is stored",
    )

    parser.add_argument(
        '--eval_path',
        type=str,
        default='../data/img/eval',
        help='the eval path to save the generated images',
    )

    parser.add_argument(
        '--eval_file',
        type=str,
        default='../data/img/eval.json',
        help='the eval file to save the eval prompts',
    )

    args = parser.parse_args()

    os.makedirs(args.eval_path, exist_ok=True)
    return args

# The transform function for dataset
def preprocess(item, transform, tokenizer, identifier=None):

    real_images = [transform(image.convert('RGB')) for image in item['real_image']]
    generated_images = [transform(image.convert('RGB')) for image in item['generated_image']]

    if identifier:
        real_prompts = [s.replace('[V]', identifier) for s in item['real_prompt']]
        generated_prompts = [re.sub(r'\s+', ' ', s.replace('[V]', '')) for s in item['generated_prompt']]

    real_prompts = tokenizer(real_prompts, max_length=tokenizer.model_max_length, padding='do_not_pad', truncation=True).input_ids
    generated_prompts = tokenizer(generated_prompts, max_length=tokenizer.model_max_length, padding='do_not_pad', truncation=True).input_ids

    return {
            'real_images': real_images,
            'real_prompts': real_prompts,
            'generated_images': generated_images,
            'generated_prompts': generated_prompts
        }

def generate_identifier(length = 3):
    # Define the length of the string sequence
    length = length

    # Define the pool of characters to choose from
    characters = string.ascii_letters

    # Generate the random string sequence
    random_sequence = ''.join(random.choice(characters) for i in range(length))

    return random_sequence

if __name__ == '__main__':
    config = parse_args()

    dataset = get_dataset(real_json=config.real_path, generated_json=config.generated_path, subject=config.subject, root_path=config.img_path)

    tokenizer = CLIPTokenizer.from_pretrained(config.model_id, subfolder="tokenizer")

    transform = transforms.Compose(
    [
        transforms.Resize((512, 512), interpolation = transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    identifier = generate_identifier(config.identifier_len)

    preprocess_fn = functools.partial(preprocess, transform=transform, tokenizer=tokenizer, identifier=identifier)

    dataset.set_transform(preprocess_fn)

    """# testing the dataset
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    index = np.random.randint(0, len(dataset), 4)
    sample_data = dataset[index]
    for i, image in enumerate(sample_data["real_images"]):
        axs[i].imshow(image.permute(1, 2, 0).numpy() / 2 + 0.5)
        axs[i].set_axis_off()
    plt.show()

    for i, prompt in enumerate(sample_data['real_prompts']):
        print(prompt)"""

    eval_list = json.load(open(config.eval_file, 'r'))
    eval_list = tokenizer([s.replace('[V]', identifier) for s in eval_list[config.subject]], max_length=tokenizer.model_max_length, padding='do_not_pad', truncation=True).input_ids





