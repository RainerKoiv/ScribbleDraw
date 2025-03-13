#from fastapi import FastAPI, UploadFile, File
#from contextlib import asynccontextmanager
import os
from PIL import Image
import time
import numpy as np
#import io
import torch
#from diffusers import AutoPipelineForImage2Image
#from diffusers.utils import make_image_grid, load_image
#from diffusers import StableDiffusionImg2ImgPipeline, AutoPipelineForImage2Image, StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
#from PIL import Image
#%pip install mediapipe
from controlnet_aux import HEDdetector
#from diffusers.utils import load_image
import time
#%pip install tensorflow
#import requests

import torch
import gc
#from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import BlipProcessor, BlipForConditionalGeneration
#from transformers import Blip2Processor, Blip2ForConditionalGeneration


# GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load BLIP model for captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Load BLIP-2 processor and model
#blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
#blip_model = Blip2ForConditionalGeneration.from_pretrained(
#    "Salesforce/blip2-opt-2.7b", torch_dtype=torch_dtype
#).to(device)

# Controlnet
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# Florence 2 large
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

hed = HEDdetector.from_pretrained('lllyasviel/Annotators')
#hed = HEDdetector.from_pretrained('control_v11p_sd15_scribble') # vajab autentimist

#url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
#image = Image.open(requests.get(url, stream=True).raw)
def run_example(task_prompt, text_input=None, image=None, processor=None, model=None, device=None, torch_dtype=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

    print(parsed_answer)
    return parsed_answer


def fix_prompt(prompt):
    if "A black and white" in prompt:
        prompt = prompt.replace("A black and white", "")
    if "on a black background." in prompt:
        prompt = prompt.replace("on a black background.", "")
    remove = ["white", "black", "drawing", "sketch", "background", "outline"]
    words = prompt.split()
    new_prompt = ""
    print(words)
    for w in words:
        if w not in remove:
            new_prompt += w + " "
    return new_prompt

#def generate_caption(image):
    """Generate a caption for a given scribble drawing (PIL image or file path)."""

    # Provide a guiding prompt
    prompt_text = "A picture of"

    inputs = blip_processor(images=image, text=prompt_text, return_tensors="pt").to(device) 
    #inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        caption = blip_model.generate(**inputs)

    caption_text = blip_processor.batch_decode(caption, skip_special_tokens=True)[0]
    
    # Filter out words
    forbidden_words = ["black", "white", "background", "drawing", "sketch", "line", "outline"]
    filtered_caption = " ".join([word for word in caption_text.split() if word.lower() not in forbidden_words])

    return filtered_caption

#def choose_style(object, style):
    styles = {
        "Realism": f"A highly detailed and realistic depiction of {object}, with accurate lighting, shadows, and textures. The colors are true-to-life, and the composition looks like a professional photograph or a classical realistic painting.",
        "Impressionism": f"A beautiful impressionist painting of {object}, with soft brushstrokes, vibrant colors, and a dreamy atmosphere. The lighting is natural, capturing a fleeting moment with artistic motion and a focus on light and color blending.",
        "Surrealism": f"A surreal and dreamlike interpretation of {object}, featuring unexpected, bizarre, and otherworldly elements. The scene is imaginative and mysterious, blending reality with fantasy in a way that defies logic.",
        "Pop Art": f"A bold and colorful pop-art version of {object}, inspired by Andy Warhol and Roy Lichtenstein. The colors are bright and saturated, with thick outlines and a comic book or advertisement-style aesthetic.",
        "Watercolor": f"A soft and artistic watercolor painting of {object}, with delicate brushstrokes, fluid color blending, and a light, airy feel. The colors bleed naturally into one another, creating a dreamy and painterly effect.",
        "Pixel Art": f"A retro pixel-art rendition of {object}, with a low-resolution 8-bit or 16-bit aesthetic. The image consists of small square pixels, giving it a nostalgic video game look with bright, limited colors.",
        "Cartoon/Comic Style": f"A fun and stylized cartoon-style illustration of {object}, with bold outlines, exaggerated proportions, and vibrant colors. The shading is cel-shaded, similar to classic animated shows or comics.",
        "Steampunk": f"A steampunk-inspired interpretation of {object}, featuring Victorian-era aesthetics combined with futuristic machinery. Gears, steam-powered devices, and brass textures give the artwork an industrial, retro-futuristic feel.",
        "Gothic": f"A dark and moody gothic illustration of {object}, with high contrast lighting, intricate Victorian-inspired details, and a sense of mystery. The atmosphere is eerie and dramatic, evoking gothic horror themes.",
        "Futurism": f"A high-tech, futuristic interpretation of {object}, with neon lights, sleek metallic surfaces, and a sense of movement. The artwork features advanced technology, cyberpunk elements, and a futuristic cityscape.",
        "Anime": f"A vibrant and expressive anime-style illustration of {object}, featuring smooth shading, large expressive eyes, and dynamic character design. The colors are bright, with detailed backgrounds and action-oriented composition, inspired by Japanese animation."
    }
    return styles.get(style)
def choose_style(object, style):
    styles = {
        "Realism": f"A highly detailed and realistic depiction of {object}, with accurate lighting, shadows, and textures. The colors are true-to-life, and the composition looks like a professional photograph or a classical realistic painting.",
        "Photorealistic CGI": f"A hyper-realistic CGI rendering of {object}, with highly detailed textures, perfect lighting, and a polished, modern 3D appearance.",
        "Impressionism": f"A beautiful impressionist painting of {object}, with soft brushstrokes, vibrant colors, and a dreamy atmosphere. The lighting is natural, capturing a fleeting moment with artistic motion and a focus on light and color blending.",
        "Surrealism": f"A surreal and dreamlike interpretation of {object}, featuring unexpected, bizarre, and otherworldly elements. The scene is imaginative and mysterious, blending reality with fantasy in a way that defies logic.",
        "Pop Art": f"A bold and colorful pop-art version of {object}, inspired by Andy Warhol and Roy Lichtenstein. The colors are bright and saturated, with thick outlines and a comic book or advertisement-style aesthetic.",
        "Pixel Art": f"A retro pixel-art rendition of {object}, with a low-resolution 8-bit or 16-bit aesthetic. The image consists of small square pixels, giving it a nostalgic video game look with bright, limited colors.",
        "Sketch & Ink Art": f"A refined high-quality sketch of {object}, with expressive linework and subtle shading. The drawing maintains a hand-drawn feel, similar to concept art or ink illustrations. Crisp, confident lines use cross-hatching or stippling for depth, resembling expert concept art or ink illustrations.",
        "Futurism": f"A high-tech, futuristic interpretation of {object}, with neon lights, sleek metallic surfaces, and a sense of movement. The artwork features advanced technology, cyberpunk elements, and a futuristic cityscape.",
        "Gothic": f"A dark and moody gothic illustration of {object}, with high contrast lighting, intricate Victorian-inspired details, and a sense of mystery. The atmosphere is eerie and dramatic, evoking gothic horror themes.",
        "Minimalism": f"A minimalist and clean depiction of {object}, with simple shapes, flat colors, and little to no extra detail, creating a modern, stylish aesthetic.",
        #"Anime": f"A vibrant and expressive anime-style illustration of {object}, featuring bold outlines, smooth shading, and dynamic composition. If the object is a character, it has large expressive eyes and stylized proportions. If it's an inanimate object, it appears with exaggerated details, vibrant colors, and a polished, hand-drawn aesthetic, inspired by Japanese animation.",
        "Anime": f"A vibrant and expressive anime-style illustration of {object}, featuring smooth shading, large expressive eyes, and dynamic character design. The colors are bright, with detailed backgrounds and action-oriented composition, inspired by Japanese animation."
    }
    return styles.get(style)

def choose_background(background):
    backgrounds = {
        #"None": "A plain, minimal background.",
        "None": "",
        #"Natural": "A background with elements of nature, evoking a peaceful and organic feel.",
        "Natural": "A natural environment with lush greenery, blue skies, and soft sunlight, evoking a peaceful and organic feel.",
        "Urban": "An urban setting with architectural elements, providing a sense of city life.",
        "Studio Lighting": "A controlled lighting environment, highlighting the subject professionally.",
        "Fantasy": "A dreamy, imaginative background with a surreal or otherworldly atmosphere."
    }
    return backgrounds.get(background)

#def choose_background(background):
    backgrounds = {
        "None": "A clean, simple background with no distractions, either plain white or transparent.",
        "Natural": "A natural environment with lush greenery, blue skies, and soft sunlight, evoking a peaceful and organic feel.",
        "Fantasy": "A magical fantasy world with glowing mushrooms, floating islands, enchanted forests, and mystical lighting.",
        "Urban": "A vibrant urban environment featuring tall skyscrapers, neon lights, and bustling streets with a cyberpunk or old-town charm.",
        "Abstract": "An abstract background with soft color gradients, geometric shapes, or artistic splashes of paint, adding a creative touch.",
        "Textured/Patterned": "A textured background resembling watercolor paper, canvas, or artistic brush strokes, adding depth to the composition.",
        "Studio Lighting": "A professional studio setting with soft, even lighting and a blurred gradient backdrop, creating a polished look."
    }
    return backgrounds.get(background)

def save_image(generated_image, path, original):
    if not os.path.exists(path):
        os.makedirs(path)
    
    if original:
        drawing_name = f"drawing_{int(time.time())}.png"
    else:
        drawing_name = f"enhanced_drawing_{int(time.time())}.png"
    file_path = os.path.join(path, drawing_name)
    generated_image.save(file_path)
    print(f"Image saved to {file_path}")

# Kasutab Florence't ja ControlNet'i
def enhance_drawing(drawing, radio, style, background, canvas_size):
    t1 = time.time()
    # Annab tausta, layerid ja composite, võta joonis
    drawing = drawing["composite"]
    image = hed(drawing, scribble=False) # scribble=True päris pildi puhul

    # Check if the user has drawn anything
    sketch = True
    info_msg = "\nTip: You can press 'Generate image' multiple times to see different results."
    object = ""
    if drawing is None or np.sum(drawing) == 0:
        info_msg = "You have not drawn anything. Generated a random scene without semantic info." + info_msg
        sketch = False
    
    # Object
    if radio == "Detect object class and generate" and sketch:
        # FLORENCE
        prompt = "<OD>"
        #prompt = "<CAPTION>"
        #prompt = "<VQA>"
        #text_input = "What is this image about?"
        #caption = run_example(prompt, text_input=text_input, image=image, processor=processor, model=model, device=device, torch_dtype=torch_dtype)
        caption = run_example(prompt, image=image, processor=processor, model=model, device=device, torch_dtype=torch_dtype)

        #extracted_caption = caption.get("<CAPTION>")
        #object = fix_prompt(extracted_caption)
        #object = caption

        object = caption.get('<OD>', {}).get('labels', [])[0]
        info_msg = f"Detected object class: {object}." + info_msg
        print(object)

    # Caption
    if radio == "Describe the scene with VLM and generate" and sketch:
        # Blip
        #caption1 = generate_caption(image)
        #print("cap1:", caption1)
        
        # Florence
        prompt = "<CAPTION>"
        caption2 = run_example(prompt, image=image, processor=processor, model=model, device=device, torch_dtype=torch_dtype)
        extracted_caption = caption2.get("<CAPTION>")
        info_msg = f"Generated scene description: '{extracted_caption}'." + info_msg
        object = fix_prompt(extracted_caption)
        print("cap2",object)

    # Nothing
    if radio == "Generate based on edges without semantic info" and sketch:
        info_msg = "Generated a random scene based on edges without semantic info." + info_msg

    style = choose_style(object, style)
    background = choose_background(background)
    prompt = style + background
    #negative_prompt="worst quality, normal quality, low quality, low res, blurry, distortion, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch, duplicate, ugly, monochrome, horror, geometry, mutation, disgusting, bad anatomy, bad proportions, bad quality, deformed, disconnected limbs, out of frame, out of focus, dehydrated, disfigured, extra arms, extra limbs, extra hands, fused fingers, gross proportions, long neck, jpeg, malformed limbs, mutated, mutated hands, mutated limbs, missing arms, missing fingers, picture frame, poorly drawn hands, poorly drawn face, collage, pixel, pixelated, grainy, color aberration, amputee, autograph, bad illustration, beyond the borders, blank background, body out of frame, boring background, branding, cut off, dismembered, disproportioned, distorted, draft, duplicated features, extra fingers, extra legs, fault, flaw, grains, hazy, identifying mark, improper scale, incorrect physiology, incorrect ratio, indistinct, kitsch, low resolution, macabre, malformed, mark, misshapen, missing hands, missing legs, mistake, morbid, mutilated, off-screen, outside the picture, poorly drawn feet, printed words, render, repellent, replicate, reproduce, revolting dimensions, script, shortened, sign, split image, squint, storyboard, tiling, trimmed, unfocused, unattractive, unnatural pose, unreal engine, unsightly, written language"
    negative_prompt = (
    "deformed, mutated, malformed, extra limbs, extra fingers, extra hands, distorted, bad proportions, "
    "disfigured, missing limbs, fused fingers, floating limbs, broken anatomy, "
    "low quality, blurry, pixelated, ugly, glitch, error, duplicate, collage, jpeg artifacts"
)
    t2 = time.time()
    new_image = pipe(prompt, image, strength=0.35, guidance_scale=7.0, num_inference_steps=30, negative_prompt=negative_prompt, controlnet_conditioning_scale=0.8, width=canvas_size[0], heigth=canvas_size[1]).images[0] #kontrolli kas strength ja guidance_scale mõjutavad, päris hea 0.7 ja 20.0 v 0.4, 15; 0.4 ja 8.5
    t3 = time.time()

    path = "images"
    original = True
    save_image(image, path, original)
    original = False
    save_image(new_image, path, original)

    print("Time taken for all steps: ", t3-t1)
    print("Time taken for image generation: ", t3-t2)
    
    torch.cuda.empty_cache()  # Clear unused memory
    gc.collect()  # Force garbage collection
    return new_image, info_msg