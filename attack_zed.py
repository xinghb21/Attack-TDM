import torch
from openai import OpenAI
from transformers import AutoFeatureExtractor, CLIPProcessor, CLIPModel, CLIPTokenizer
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers import StableDiffusionPipeline
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
from transformers.models.clip.image_processing_clip import CLIPImageProcessor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clip_dir = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
clip_model = CLIPModel.from_pretrained(clip_dir).to(device)
preprocess = CLIPProcessor.from_pretrained(clip_dir)
tokenizer = CLIPTokenizer.from_pretrained(clip_dir)

# Initialize models
diff_dir = "/data1/zhaoed/models/stable-diffusion-2-1/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6"
checker_dir = "/data1/zhaoed/models/stable-diffusion-safety-checker/models--CompVis--stable-diffusion-safety-checker/snapshots/cb41f3a270d63d454d385fc2e4f571c487c253c5"
diffusion_pipeline = StableDiffusionPipeline.from_pretrained(diff_dir).to(device)
checker = StableDiffusionSafetyChecker.from_pretrained(checker_dir)
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(checker_dir)

# Define parameters
m = 10  # maximum text length
n = 10  # number of inner iterations
lambda_ = 0.5  # weight for cosine similarity in loss
alpha = 0.01  # step size
r = random.random()  # random number'

def get_text_embedding(inputs):
    '''
    inputs: a str or a tokenized Tensor (input_ids)
    '''
    if isinstance(inputs, str):
        input_ids = tokenizer(inputs, padding='max_length', max_length=tokenizer.model_max_length,
                            truncation=True, return_tensors='pt').input_ids.to(device)[0]
    else:
        input_ids = inputs                    
    input_ids = input_ids.unsqueeze(0)
    pooled_output = diffusion_pipeline.text_encoder(input_ids)[1]
    proj_emb = clip_model.text_projection(pooled_output)
    return proj_emb

cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
loss_fn = lambda x, y: 1 - cos(x.view(-1), y.view(-1))

image = Image.open(f'OIP.jpg')
inputs = preprocess(images=[image], return_tensors="pt").to(device)
image_embeds = clip_model.get_image_features(**inputs).detach()
print("Image embeddings:", image_embeds.shape)
text_embeds = get_text_embedding("A photo of a cat").detach()
loss = loss_fn(image_embeds, text_embeds)
print(loss)

client = OpenAI(
    base_url="https://api.gptsapi.net/v1",
    api_key="sk-7gh0c134910ee959aa6cf2d36d027427de0f4719654Tu1yM"
)

# Initialize prompt
prompt = "A photo of a cat"
prompt_embeds = get_text_embedding(prompt).detach()

# Start the optimization loop
for t in range(m):
    # Generate k possible words using GPT
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Generate ONLY 10 words that could follow the prompt: " + prompt + ". Respond with one word per line and without linemark."},
        ],
        max_tokens=50,
        temperature=1.0
    )
    k_words = response.choices[0].message.content.split("\n")
    print("Generated words:", k_words)

    # diffusion_pipeline = diffusion_pipeline.to('cpu')

    # Compute token embeddings for k words
    k_word_embeddings = [diffusion_pipeline.encode_prompt(word, device, 1, True)[0].to(device) for word in k_words]
    print("Encoded words:", k_word_embeddings[0].shape)

    # Compute token embedding of the current prompt
    prompt_embedding = diffusion_pipeline.encode_prompt(prompt, device, 1, True)[0].to(device)
    print("Prompt embedding:", prompt_embedding.shape)

    # Initialize latent code
    z = torch.randn(1, 4, 64, 64).to(device)

    # Randomly initialize words embedding τ[x,y]
    random_embeddings = torch.randn_like(k_word_embeddings[0]).to(device).detach()
    random_embeddings.requires_grad = True

    optim = torch.optim.Adam([random_embeddings], lr=1e-5)

    for d in range(n):
        # Concatenate current token embedding
        current_token = prompt_embedding
        print("Current token:", current_token.shape)

        # Generate image
        diffusion_pipeline = diffusion_pipeline.to(device)
        generator = torch.Generator(device).manual_seed(0)
        generated_image = diffusion_pipeline(prompt_embeds=current_token, z=z, generator=generator, num_inference_steps=70).images[0]
        inputs = preprocess(images=[generated_image], return_tensors="pt").to(device)
        generated_image.save(f'generated_image_{t}_{d}.png')

        # Compute τ[cn]: find the closest embedding in k_word_embeddings to τ[x]
        distance = torch.Tensor([torch.cosine_similarity(random_embeddings.reshape(-1), k_word_embedding.reshape(-1), dim=0) for k_word_embedding in k_word_embeddings])
        closest_idx = torch.argmin(distance)
        closest_embedding = k_word_embeddings[closest_idx]

        # Compute loss
        image_embeds = clip_model.get_image_features(**inputs).detach()
        image_loss = loss_fn(image_embeds, prompt_embeds)
        loss = image_loss + lambda_ * torch.cosine_similarity(random_embeddings.reshape(-1), closest_embedding.reshape(-1), dim=0)

        # Update embeddings
        loss = loss.to(device)
        optim.zero_grad()
        loss.backward()
        optim.step()
        # grad = random_embeddings.grad
        # random_embeddings = random_embeddings.clone().detach()
        # random_embeddings = random_embeddings + r * alpha * torch.sign(grad)

    # Find the closest candidate word and update the prompt
    closest_word = k_words[closest_idx]

    # Update prompt
    prompt += " " + closest_word

print("Final prompt:", prompt)
