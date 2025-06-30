# Imports and installations
import os
import time
import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
#%pip install mediapipe
from controlnet_aux import HEDdetector
import time
#%pip install tensorflow
import gc
from transformers import AutoProcessor, AutoModelForCausalLM


# GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Controlnet
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# Florence 2 large
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

hed = HEDdetector.from_pretrained('lllyasviel/Annotators')


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

# Remove unwanted words from the auto-generated prompt
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
        "Anime": f"A vibrant and expressive anime-style illustration of {object}, featuring smooth shading, large expressive eyes, and dynamic character design. The colors are bright, with detailed backgrounds and action-oriented composition, inspired by Japanese animation.",
        "Fantasy": f"A magical and enchanting fantasy illustration of {object}, set in a mystical world with mythical creatures, vibrant colors, and a sense of wonder. The scene is filled with fantastical elements like glowing lights, ethereal landscapes, and whimsical details."
    }
    return styles.get(style)

def choose_background(background):
    backgrounds = {
        #"None": "A plain, minimal background.",
        "None": "",
        "Natural": "A natural environment with lush greenery, blue skies, and soft sunlight, evoking a peaceful and organic feel.",
        "Urban": "An urban setting with architectural elements, providing a sense of city life.",
        "Studio Lighting": "A controlled lighting environment, highlighting the subject professionally.",
        "Fantasy": "A dreamy, imaginative background with a surreal or otherworldly atmosphere."
    }
    return backgrounds.get(background)


def get_prompt_from_info_msg(info_msg):
    prompt = info_msg.split(".")[0]
    return prompt

def save_image(image, new_image, output_folder, output_file, info_msg):
    current_time = int(time.time())

    # Path to save images
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Path to save prompts in a text file
    prompt_path = os.path.join(output_folder, output_file)
    if not os.path.exists(prompt_path):
        with open(prompt_path, "w") as f:
            f.write("Generated prompts:\n")
    
    # Save the prompt in the text file
    info_msg = get_prompt_from_info_msg(info_msg)
    with open(prompt_path, "a") as f:
        f.write(f"{info_msg} - {str(current_time)}\n")
    
    # Save the images
    drawing_name = f"drawing_{(current_time)}.png"
    picture_name = f"enhanced_drawing_{current_time}.png"
    drawing_path = os.path.join(output_folder, drawing_name)
    image.save(drawing_path)
    print(f"Drawing saved to {drawing_path}")
    picture_path = os.path.join(output_folder, picture_name)
    new_image.save(picture_path)
    print(f"Enhanced drawing saved to {picture_path}")


# MAIN FUNCTION
# Uses Florence-2-large, ControlNet and Stable Diffusion
def enhance_drawing(sketchpad, radio, style, background, canvas_size, lang):
    t1 = time.time()
    # Gives background, layers and composite, take composite
    drawing = sketchpad["composite"]
 
    image = hed(drawing, scribble=False) # scribble=True if input is real image/picture

    # Check if the user has drawn anything
    sketch = True
    info_msg1_eng = "Selected style: " + style + ".\n"
    info_msg2_eng = "\nTip: If you see floating lines, try coloring the object in."
    info_msg1_est = "Valitud stiil: " + style + ".\n"
    info_msg2_est = "\nVihje: Kui näed \"hõljuvaid\" jooni, proovi joonistatud objekti seest värviga täita."
    object = ""
    if drawing is None or np.sum(drawing) == 0:
        if lang == "en":
            info_msg = info_msg1_eng + "You have not drawn anything. Generated a random scene with the selected style." + info_msg2_eng
        else:
            info_msg = info_msg1_est + "Sa pole midagi joonistanud. Genereerisin juhusliku stseeni valitud stiilis." + info_msg2_est
        sketch = False
    
    # Object
    if radio == "Detect object class and generate" and sketch:
        # Florence-2-large
        prompt = "<OD>"
        caption = run_example(prompt, image=image, processor=processor, model=model, device=device, torch_dtype=torch_dtype)
        object = caption.get('<OD>', {}).get('labels', [])[0]
        if lang == "en":
            info_msg = info_msg1_eng + f"Detected object class: {object}" + info_msg2_eng
        else:
            info_msg = info_msg1_est + f"Tuvastatud objekti klass: {object}" + info_msg2_est

    # Caption
    if radio == "Describe the scene with VLM and generate" and sketch:
        # Florence-2-large
        prompt = "<CAPTION>"
        caption2 = run_example(prompt, image=image, processor=processor, model=model, device=device, torch_dtype=torch_dtype)
        extracted_caption = caption2.get("<CAPTION>")
        if lang == "en":
            info_msg = info_msg1_eng + f"AI interpretation of your drawing: '{extracted_caption}'" + info_msg2_eng
        else:
            info_msg = info_msg1_est + f"AI tõlgendus kritseldusele: '{extracted_caption}'" + info_msg2_est
            
        object = fix_prompt(extracted_caption)

    # Nothing
    if radio == "Generate based on edges without semantic info" and sketch:
        if lang == "en":
            info_msg = info_msg1_eng + "Generated a random scene based on edges without semantic info." + info_msg2_eng
        else:
            info_msg = info_msg1_est + "Joonistasin ainult jooni arvestades juhusliku stseeni ilma semantilise infota." + info_msg2_est

    style = choose_style(object, style)
    background = choose_background(background)
    prompt = style + background
    
    negative_prompt = (
    "deformed, mutated, malformed, extra limbs, extra fingers, extra hands, distorted, bad proportions, "
    "disfigured, missing limbs, fused fingers, floating limbs, broken anatomy, "
    "low quality, blurry, pixelated, ugly, watermark, error, duplicate, collage, jpeg artifacts"
    )
    t2 = time.time()
    new_image = pipe(prompt, image, strength=0.35, guidance_scale=7.5, num_inference_steps=40, negative_prompt=negative_prompt, controlnet_conditioning_scale=0.9, width=canvas_size[0], heigth=canvas_size[1]).images[0]
    t3 = time.time()

    output_folder = "images" # Change this!
    output_file = "prompts.txt"
    save_image(image, new_image, output_folder, output_file, info_msg)

    print("Time taken for all steps: ", t3-t1)
    print("Time taken for image generation: ", t3-t2)
    
    torch.cuda.empty_cache()  # Clear unused memory
    gc.collect()  # Force garbage collection
    return new_image, info_msg