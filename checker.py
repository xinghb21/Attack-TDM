from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", revision='fp16', use_auth_token=True)
from PIL import Image
import torch
import numpy as np
from transformers import AutoFeatureExtractor

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
safety_feature_extractor = AutoFeatureExtractor.from_pretrained("CompVis/stable-diffusion-safety-checker")

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

# pipe = pipe.to('mps')
# generator = torch.Generator("mps").manual_seed(0)
# image = pipe(['a photo of a nude girl'], generator=generator, num_inference_steps=70).images[0]
# image.save(f'image.png')
# exit()
def check_image(image_path):
    image = Image.open(image_path)
    image = np.array(image)
    safety_input = safety_feature_extractor(numpy_to_pil(image), return_tensors='pt')
    output_image, has_nsfw_concept = checker(images=image, clip_input=safety_input.pixel_values)
    output_image = numpy_to_pil(output_image)
    output_image[0].save('output.png')
    print(has_nsfw_concept)

check_image('yellow.jpeg')