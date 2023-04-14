import argparse, os, random, string, json, math
from tqdm.auto import tqdm
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from utils.DreamBooth_Dataset import get_dataset
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import functools
from torchvision import transforms
import re
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from accelerate import Accelerator
import torch.nn.functional as F
from utils.tools import test_generated_imgs

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
        default=5,
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
        '--log_path',
        type=str,
        default='../log/logs',
        help='the eval path to save the logs (usually tensorboard logs)',
    )

    parser.add_argument(
        '--eval_file',
        type=str,
        default='../data/eval.json',
        help='the eval file to save the eval prompts',
    )

    parser.add_argument(
        '--save_path',
        type=str,
        default='../log/saved_models',
        help='the path to save the model checkpoints',
    )

    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=8,
        help='the batch size for training',
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=5e-06,
        help='the batch size for training',
    )

    parser.add_argument(
        '--num_train_epoch',
        type=int,
        default=100,
        help='the batch size for training',
    )

    parser.add_argument(
        '--gradient_accumlation_steps',
        type=int,
        default=1,
        help='the batch size for training',
    )

    parser.add_argument(
        '--eval_every_steps',
        type=int,
        default=50,
        help='the batch size for training',
    )


    args = parser.parse_args()

    os.makedirs(args.eval_path, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    return args

# The transform function for dataset
def preprocess(item, transform, tokenizer, identifier=None):

    real_images = [transform(image.convert('RGB')) for image in item['real_image']]
    generated_images = [transform(image.convert('RGB')) for image in item['generated_image']]

    if identifier:
        generated_prompts = [s.replace('[V]', identifier) for s in item['generated_prompt']]
        real_prompts = [re.sub(r'\s+', ' ', s.replace('[V]', '')) for s in item['real_prompt']]

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

def collate_fn(item):
    real_images = torch.stack([img['real_images'] for img in item]).to(memory_format=torch.contiguous_format).float()
    generated_images = torch.stack([img['generated_images'] for img in item]).to(memory_format=torch.contiguous_format).float()

    real_prompts = [prompt['real_prompts'] for prompt in item]
    generated_prompts = [prompt['generated_prompts'] for prompt in item]

    real_prompts = tokenizer.pad({"input_ids": real_prompts}, padding=True, return_tensors="pt")
    generated_prompts = tokenizer.pad({"input_ids": generated_prompts}, padding=True, return_tensors="pt")

    return{
            'real_images': real_images,
            'real_prompts': real_prompts.input_ids,
            'generated_images': generated_images,
            'generated_prompts': generated_prompts.input_ids
    }

def eval(config, epoch, promts, text_encoder, vae, unet, device='cuda', repo=None):
    pipeline = StableDiffusionPipeline(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=PNDMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
        ),
        safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
        feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
    )

    n_samples, scale, steps = 1, 7.5, 50
    if device is not None:
        pipeline.to(device)

    for prompt in promts:
        os.makedirs(f"{config.eval_path}/{config.subject}", exist_ok=True)
        images = pipeline(prompt, guidance_scale=scale, num_inference_steps=steps).images

        for idx, im in enumerate(images):
            im.save(f"{config.eval_path}/{config.subject}/{prompt}_{epoch}_{idx:02}.png")

    pipeline.save_pretrained(os.path.join(config.save_path, config.subject, f'saved_model_{epoch}'))
    del pipeline

    if repo is not None:
        pipeline.save_pretrained(os.path.join(config.output_dir, 'saved_model'))
        repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=False, auto_lfs_prune=True)

if __name__ == '__main__':
    config = parse_args()

    # prepare the dataset
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
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True,
                                                   collate_fn=collate_fn)

    """#test
    for step, batch in enumerate(train_dataloader):
        test_generated_imgs(batch['generated_images'])"""

    # read the eval list from json file
    eval_list = json.load(open(config.eval_file, 'r'))
    eval_list = [s.replace('[V]', identifier) for s in eval_list[config.subject]]

    # Load the pre-trained model from checkpoints
    text_encoder = CLIPTextModel.from_pretrained(config.model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config.model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config.model_id, subfolder="unet")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(True)
    unet.requires_grad_(True)
    unet.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW([
        {'params': unet.parameters()},
        {'params': text_encoder.parameters()}
    ], lr=config.lr, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-08)

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )

    # start training
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumlation_steps,
        mixed_precision='fp16',  # whether to use a mixed precision (fp16 & bf16)
        log_with='tensorboard',  # usually go to the tensorborad
        logging_dir=os.path.join(config.log_path, config.subject)
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("train_example")

    vae.to(accelerator.device)

    unet, text_encoder, optimizer, train_dataloader = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader
    )

    global_step = 0
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumlation_steps)

    for epoch in range(config.num_train_epoch):
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):

            unet.train()
            text_encoder.train()

            # accumulate the gradient to simulate batch gradient descent
            with accelerator.accumulate(unet):
                # Predict the noise residual
                real_images = batch['real_images']
                generated_images = batch['generated_images']

                # compute the latents using the pretrained vae
                real_latents = vae.encode(real_images).latent_dist.sample()
                real_latents = 0.18215 * real_latents
                generated_latents = vae.encode(generated_images).latent_dist.sample()
                generated_latents = 0.18215 * generated_latents

                # generate the noise for reparameterized trick
                real_noise = torch.randn_like(real_latents)
                generated_noise = torch.randn_like(generated_latents)
                bsz = real_latents.shape[0]

                # get the noisy latents using scheduler
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=real_latents.device).long()
                noisy_real_latents = noise_scheduler.add_noise(real_latents, real_noise, timesteps)
                noisy_generated_latents = noise_scheduler.add_noise(generated_latents, generated_noise, timesteps)

                # predict the original image under the condition of the caption's embedding
                real_encoder_hiddenstates = text_encoder(batch['real_prompts'])[0]
                real_noisy_pred = unet(noisy_real_latents, timesteps, real_encoder_hiddenstates).sample
                generated_encoder_hiddenstates = text_encoder(batch['generated_prompts'])[0]
                generated_noisy_pred = unet(noisy_generated_latents, timesteps, generated_encoder_hiddenstates).sample

                loss = F.mse_loss(real_noisy_pred, real_noise, reduction="mean") + F.mse_loss(generated_noisy_pred, generated_noise, reduction="mean")
                avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                train_loss += avg_loss.item() / config.gradient_accumlation_steps

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(list(unet.parameters()) + list(text_encoder.parameters()), 1.0)

                optimizer.step()
                #lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss = 0.0

                logs = {"loss": loss.detach().item(), "lr": config.lr, "step": global_step}
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)

            if accelerator.is_main_process:
                # eval the image and uplaod the model to the repo
                if (global_step+1) % config.eval_every_steps == 0:
                    eval(config, global_step + 1, eval_list, text_encoder, vae, unet, device=accelerator.device)