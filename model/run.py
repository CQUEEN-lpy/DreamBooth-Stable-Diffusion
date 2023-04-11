import argparse, os, json
from tqdm.auto import tqdm
from utils.DreamBooth_Dataset import get_dataset
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import functools
from torchvision import transforms
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

    args = parser.parse_args()

    os.makedirs(args.eval_path, exist_ok=True)
    return args

# The transform function for dataset
def preprocess(item, transform, tokenizer):

    real_images = [transform(image.convert('RGB')) for image in item['real_image']]
    generated_images = [transform(image.convert('RGB')) for image in item['generated_image']]

    real_prompts = tokenizer(item['real_prompt'])
    generated_prompts = tokenizer(item['generated_prompt'])

    return {
        {
            'real_images': real_images,
            'real_prompts': generated_images,
            'generated_images': real_prompts,
            'generated_prompts': generated_prompts
        }
    }


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

    preprocess_fn = functools.partial(preprocess, transform=transform, tokenizer=tokenizer)

    dataset.set_transform(transform)

    print(dataset[1])





