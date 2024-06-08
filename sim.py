from transformers import CLIPModel, CLIPProcessor
import torch
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clip_dir = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
clip_model = CLIPModel.from_pretrained(clip_dir, torch_dtype=torch.float16).to(device)
preprocess = CLIPProcessor.from_pretrained(clip_dir, torch_dtype=torch.float16)

def sim_score(generated_image, init_prompt):
    inputs = preprocess(images=generated_image, text=[init_prompt], return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    image_loss = logits_per_image[0]
    return image_loss.item()

if __name__ == "__main__":
    image_dir = "./figures/tokenizer/dog/"
    prompt = "A photo of a dog"
    scores = []
    for i in range (10):
        image = Image.open(image_dir + str(i) + "/19.png")
        score = sim_score(image, prompt)
        scores.append(score)
    print(scores)