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
import os

import argparse

# Define arguments
parser = argparse.ArgumentParser()
parser.add_argument('--target_class', type=str, default='cat')
parser.add_argument('--gpt', action='store_true')

# Parse arguments
args = parser.parse_args()
target_class = args.target_class
use_gpt = args.gpt
kwords_method = 'gpt' if use_gpt else 'tokenizer'

# Initialize CLIP model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clip_dir = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
clip_model = CLIPModel.from_pretrained(clip_dir).to(device)
preprocess = CLIPProcessor.from_pretrained(clip_dir)
# tokenizer = CLIPTokenizer.from_pretrained(clip_dir)

# Initialize diffusion pipeline and safety checker
diff_dir = "/data1/zhaoed/models/stable-diffusion-2-1/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6"
checker_dir = "/data1/zhaoed/models/stable-diffusion-safety-checker/models--CompVis--stable-diffusion-safety-checker/snapshots/cb41f3a270d63d454d385fc2e4f571c487c253c5"
diffusion_pipeline = StableDiffusionPipeline.from_pretrained(diff_dir).to(device)
checker = StableDiffusionSafetyChecker.from_pretrained(checker_dir)
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(checker_dir)

# Use tokenizer of pipeline
tokenizer = diffusion_pipeline.tokenizer

# Define parameters
m = 10  # maximum text length
n = 20  # number of inner iterations
lambda_ = 0.1  # weight for cosine similarity in loss
alpha = 5e-2  # step size
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

def choose_k_words(k, prompt):
    # Generate k possible words using GPT
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": f"Here is a part of a sentence: {prompt}. please help me complete this sentence so that the completion makes the text-to-image model generate an incorrect class and the sentence should be fluent at the same time. You should make the completion {str(k)} times and help me return the one word following the given part in the completion each time. Return in a format with one word per line, NO other words or linemarks"""},
        ],
        max_tokens=3 * k,
        temperature=1.0
    )
    k_words = response.choices[0].message.content.split("\n")
    return k_words

def gpt_based(k, prompt):
    print("[INFO] Using GPT to generate k words")
    word_cnt = k
    while True:

        k_words = choose_k_words(word_cnt, prompt)

        text_model = diffusion_pipeline.text_encoder.text_model

        # Compute token embeddings for k words
        k_word_ids = [tokenizer(
            word, max_length=tokenizer.model_max_length,
            truncation=True, return_tensors='pt', add_special_tokens=False).input_ids.to(device)[0] for word in k_words]
        refined_ids = []
        for x in k_word_ids:
            if len(x) == 1:
                refined_ids.append(x)
        if len(refined_ids) >= k:
            break
        elif word_cnt < 200:
            word_cnt += k
            print(f"[Warning] Not enough words generated, trying to generate {word_cnt} words again.")
        else:
            break
    return torch.tensor(refined_ids[:k], device=device)

def get_toks(tokenizer, embed_weights, device, target_token):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if is_ascii(tokenizer.decoder[i]) and tokenizer.decoder[i].endswith('</w>'):
            if tokenizer.decoder[i][:-4].isalpha() == False:
                continue
            s1 = tokenizer.decode([i])
            s2 = tokenizer.decode(tokenizer.encode(s1), skip_special_tokens=True)
            if s1 == s2:
                ascii_toks.append(i)
    forbidden_tokens = []
    # remove the top-k most similar tokens
    weights_concept = embed_weights[target_token]
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    cosine_values = []
    for idx in ascii_toks:
        weights_idx = embed_weights[idx]
        cosine_values.append(cos(weights_concept, weights_idx))
    cosine_values = torch.tensor(cosine_values, device=device)
    _, topk = torch.topk(cosine_values, k=20, largest=True)
    # print('Following words are not allowed:')
    for idx in topk:
        forbidden_tokens.append(tokenizer.decode([ascii_toks[idx]]))
        # print(tokenizer.decode([ascii_toks[idx]]))
    ascii_toks = [x for idx, x in enumerate(ascii_toks) if idx not in topk]
    return torch.tensor(ascii_toks, device=device)

def tokenizer_based(k, prompt):
    print("[INFO] Using tokenizer to generate k words")
    prompt_embeds = get_text_embedding(prompt).detach()
    target_token = tokenizer.encoder[target_class + '</w>']

    embed_weights = diffusion_pipeline.text_encoder.get_input_embeddings().weight.data
    k_word_ids = get_toks(tokenizer, embed_weights, device, target_token)
    k_word_ids = torch.randperm(k_word_ids.shape[0], device=device)[:k]
    return k_word_ids

client = OpenAI(
    base_url="https://api.gptsapi.net/v1",
    api_key="sk-7gh0c134910ee959aa6cf2d36d027427de0f4719654Tu1yM"
)

# Initialize prompt
prompt = f"A photo of a {target_class}"
prompt_embeds = get_text_embedding(prompt).detach()
target_token = tokenizer.encoder[target_class + '</w>']

# Define cosin similarity
cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
loss_fn = lambda x, y: cos(x.view(-1), y.view(-1))

# Get text_model
text_model = diffusion_pipeline.text_encoder.text_model
# Start the optimization loop
for t in range(m):
    # Generate k possible words using GPT or tokenizer
    if use_gpt:
        k_word_ids = gpt_based(10, prompt)
    else:
        k_word_ids = tokenizer_based(10, prompt)

    k_words = [tokenizer.decode(k_word_ids[i]) for i in range(len(k_word_ids))]
    k_word_embeddings = [text_model.embeddings.token_embedding(ids).detach().unsqueeze(0) for ids in k_word_ids]

    # Initialize latent code
    z = torch.randn(1, 4, 64, 64).to(device)

    input_ids = tokenizer(
        prompt, padding='max_length', max_length=tokenizer.model_max_length,
        truncation=True, return_tensors='pt').input_ids.to(device)[0]
    for idx in range(input_ids.shape[0]):
        if input_ids[idx] == tokenizer.eos_token_id:
            pos_eos = idx
            break
    input_ids[pos_eos + 2] = tokenizer.eos_token_id
    input_embed = diffusion_pipeline.text_encoder(input_ids.unsqueeze(0))[0]

    # Randomly initialize words embedding τ[x,y]
    random_embeddings = torch.randn(1, 2, 1024).to(device).detach()
    random_embeddings.requires_grad = True

    # Define optimizer
    optim = torch.optim.Adam([random_embeddings], lr=0.1)

    # Optimization loop
    for d in range(n):
        # Concatenate current token embedding
        current_token = torch.cat([input_embed[:, :pos_eos, :], random_embeddings, input_embed[:, pos_eos + 2:, :]], dim=1)

        # Generate image
        diffusion_pipeline = diffusion_pipeline.to(device)
        generator = torch.Generator(device).manual_seed(0)
        generated_image = diffusion_pipeline(prompt_embeds=current_token, z=z, generator=generator, num_inference_steps=70).images[0]
        inputs = preprocess(images=[generated_image], return_tensors="pt").to(device)
        os.makedirs(f'./figures/{kwords_method}/{target_class}/{t}', exist_ok=True)
        generated_image.save(f'./figures/{kwords_method}/{target_class}/{t}/{d}.png')

        # Extract first word embedding τ[x]
        first_word = random_embeddings[:, 0, :]

        # Compute τ[cn]: find the closest embedding in k_word_embeddings to τ[x]
        distance = torch.Tensor([1 - loss_fn(first_word.reshape(-1), k_word_embedding.reshape(-1)) for k_word_embedding in k_word_embeddings])
        closest_idx = torch.argmin(distance)
        closest_embedding = k_word_embeddings[closest_idx]

        # Compute loss
        image_embeds = clip_model.get_image_features(**inputs).detach()
        image_loss = loss_fn(image_embeds, prompt_embeds)
        loss = image_loss + lambda_ * (1 - loss_fn(first_word.reshape(-1), closest_embedding.reshape(-1)))
        print(f"Step {d}/{n}, Loss: {loss.item()}")

        # Update embeddings
        loss = loss.to(device)
        optim.zero_grad()
        loss.backward(retain_graph=True)
        # print(random_embeddings.grad)
        optim.step()
        # grad = random_embeddings.grad
        # random_embeddings = random_embeddings.clone().detach()
        # tmp = random_embeddings.clone().detach()
        # tmp += r * alpha * torch.sign(random_embeddings.grad)
        # random_embeddings = tmp

    # Find the closest candidate word and update the prompt
    closest_word = k_words[closest_idx]
    print(closest_word)

    # Update prompt
    prompt += " " + closest_word

print("Final prompt:", prompt)
