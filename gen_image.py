import torch
from diffusers import StableDiffusionPipeline
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

diff_dir = "/data1/zhaoed/models/stable-diffusion-2-1/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6"
diffusion_pipeline = StableDiffusionPipeline.from_pretrained(diff_dir, revision="fp16", torch_dtype=torch.float16).to(device)
tokenizer = diffusion_pipeline.tokenizer

prompts = [
    "A photo of a cat tracking bears",
    "A photo of a cat drawing a flower",
    "A photo of a cat bringing you beaded item",
    "A photo of a cat serving boomershot, a drink made from a mixture of vodka",
    "A photo of a cat performing karbois, a traditional dance of the Khasi people",
    "A photo of a cat (the one on the left) and a dog",
    "A photo of a cat with several people, including a panda",
    "A photo of a cat paw that looks very calming",
    "A photo of a cat looked soft from below",
    "A photo of a cat covering half Anna Politans' face",
    "A photo of a cat chasing dogs",
    "A photo of a cat teaching horses",
    "A photo of a cat running as dogs",
    "A photo of a cat holding cat ear cookies sold online",
    "A photo of a cat meeting Spider Bot made in Rasa",
    "A photo of a cat drawing car designs sketch",
    "A photo of a cat helping scientologist to clean floor, taken by a scientologist",
    "A photo of a cat sharing fish pie",
    "A photo of a cat searching vaclum cleaner",
    "A photo of a car has emerald fobs",
    "A photo of a car and rail",
    "A photo of a car behind Santa and a horse",
    "A photo of a car crossing under part ot Icela's new bridge",
    "A photo of a car bearing California license",
    "A photo of a car coming face2face with two bicycles",
    "A photo of a car behind bars",
    "A photo of a car on one way towards Manhack",
    "A photo of a car rolling straight down rail",
    "A photo of a dog playing tennis with Roger",
    "A photo of a dog meeting Etsimo, a robotic dog",
    "A photo of a bear and horse statue",
    "A photo of a bear of Chara Trailer Supplier",
    "A photo of a bear wearing sliplot, a traditional Korean hat",
    "A photo of a bird flying fast caught wind mill",
    "A photo of a bird in space circular pendant",
    "A photo of a bird suffering cat",
    "A photo of a bird running back",
    "A photo of a bird building stick lodge",
    "A photo of a bird heading deep beneath Lake Tangier in the Gale Crater",
    "A photo of a bird with my corgiwatching hat on",
    "A photo of a bird during Christmas Island Open Ocean Race at China Southern Sports University",
    "A photo of a bird lying beneath heavy deities",
    "A photo of a bird hunting moth",
    "A photo of a bird working inside Google",
    "A photo of a bird flying over two vehicles along road"
]

def gen(prompt):
    z = torch.randn(1, 4, 64, 64).to(device)
    input_ids = tokenizer(
        prompt, padding='max_length', max_length=tokenizer.model_max_length,
        truncation=True, return_tensors='pt').input_ids.to(device)[0]
    input_embed = diffusion_pipeline.text_encoder(input_ids.unsqueeze(0))[0]
    generator = torch.Generator(device).manual_seed(0)
    generated_image = diffusion_pipeline(prompt_embeds=input_embed, z=z, generator=generator, num_inference_steps=70).images[0]
    return generated_image

if __name__ == "__main__":
    save_dir = "./figures/test/"
    for i in range (len(prompts)):
        print(f"{i} / {len(prompts)}")
        prompt = prompts[i]
        image = gen(prompt)
        os.makedirs(save_dir, exist_ok=True)
        image.save(save_dir + str(i) + ".png")
        print(prompt)