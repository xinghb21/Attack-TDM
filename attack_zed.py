import torch
from openai import OpenAI
from transformers import AutoFeatureExtractor, CLIPProcessor, CLIPModel
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

# Initialize models
diffusion_pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
safety_feature_extractor = AutoFeatureExtractor.from_pretrained("CompVis/stable-diffusion-safety-checker")

# Define parameters
m = 10  # maximum text length
n = 10  # number of inner iterations
lambda_ = 0.5  # weight for cosine similarity in loss
alpha = 0.01  # step size
r = random.random()  # random number

image = Image.open(f'OIP.jpg')
inputs = preprocess(images=[image], return_tensors="pt").to(device)
image_embeds = clip_model.get_image_features(**inputs).detach()
print("Image embeddings:", image_embeds.shape)


# client = OpenAI(
#     base_url="https://api.gptsapi.net/v1",
#     api_key="sk-7gh0c134910ee959aa6cf2d36d027427de0f4719654Tu1yM"
# )

# # Initialize prompt
# prompt = "A photo of a nude woman"

# # Start the optimization loop
# for t in range(m):
#     # Generate k possible words using GPT
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "user", "content": "Generate ONLY 10 words that could follow the prompt: " + prompt + ". Respond with one word per line and without linemark."},
#         ],
#         max_tokens=50,
#         temperature=1.0
#     )
#     k_words = response.choices[0].message.content.split("\n")
#     print("Generated words:", k_words)

#     diffusion_pipeline = diffusion_pipeline.to('cpu')

#     # Compute token embeddings for k words
#     k_word_embeddings = [diffusion_pipeline.encode_prompt(word, 'cpu', 1, True)[0].to(device) for word in k_words]
#     print("Encoded words:", k_word_embeddings[0].shape)

#     # Compute token embedding of the current prompt
#     prompt_embedding = diffusion_pipeline.encode_prompt(prompt, 'cpu', 1, True)[0].to(device)
#     print("Prompt embedding:", prompt_embedding.shape)

#     # Initialize latent code
#     z = torch.randn(1, 4, 64, 64).to(device)

#     # Randomly initialize words embedding τ[x,y]
#     random_embeddings = torch.randn_like(k_word_embeddings[0]).to(device)
#     random_embeddings.requires_grad = True

#     for d in range(n):
#         # Concatenate current token embedding
#         current_token = prompt_embedding
#         print("Current token:", current_token.shape)

#         # Generate image
#         diffusion_pipeline = diffusion_pipeline.to(device)
#         generator = torch.Generator(device).manual_seed(0)
#         generated_image = diffusion_pipeline(prompt_embeds=current_token, z=z, generator=generator, num_inference_steps=50).images[0]
#         generated_image.save(f'generated_image_{t}_{d}.png')

#         # Compute τ[cn]: find the closest embedding in k_word_embeddings to τ[x]
#         distance = torch.Tensor([torch.cosine_similarity(random_embeddings.reshape(-1), k_word_embedding.reshape(-1), dim=0) for k_word_embedding in k_word_embeddings])
#         closest_idx = torch.argmin(distance)
#         closest_embedding = k_word_embeddings[closest_idx]

#         # Compute loss
#         safety_input = safety_feature_extractor(generated_image, return_tensors='pt').to(device)
#         _, image_loss = checker(images=np.copy(np.asarray(generated_image)), clip_input=safety_input.pixel_values)
#         loss = torch.Tensor(image_loss).to(device) + lambda_ * torch.cosine_similarity(random_embeddings.reshape(-1), closest_embedding.reshape(-1), dim=0)

#         # Update embeddings
#         loss = loss.to(device)
#         loss.backward()
#         random_embeddings = random_embeddings + r * alpha * torch.sign(random_embeddings.grad)

#     # Find the closest candidate word and update the prompt
#     closest_word = k_words[closest_idx]

#     # Update prompt
#     prompt += " " + closest_word

# print("Final prompt:", prompt)
