import backend as backend
import gradio as gr
import numpy as np

def enhance_image(sketchpad, radio, style, background):
    canvas_size = (768, 512) # 768x512 pixel canvas size
    enhanced_image, message = backend.enhance_drawing(sketchpad, radio, style, background, canvas_size) # enhance_drawing function from backend.py
    return enhanced_image, message 


custom_css = """
/* Toolbar buttons */
button[title="View in full screen"],
button[title="Exit full screen"],
button[title="Clear canvas"], 
button[title="Undo"], 
button[title="Redo"],
button[title="Erase button"],
button[title="Draw button"] {
    background-color: orange !important;
    width: 40px !important;
    height: 40px !important;
    font-size: 20px !important;
    color: black !important;
}

button[title="Transform button"] {
    display: none !important;
}

button[title="Draw button"] {
    margin-right: 15px !important;
}

.submit-button {
    background-color: orange !important;
    color: black !important;
    font-size: 20px !important;
}

.custom-markdown h1 {
    font-size: 32px !important;
}
.custom-markdown h2 {
    font-size: 28px !important;
}
.custom-markdown p {
    font-size: 20px !important;
}

.custom-markdown-footer h1 {
    font-size: 20px !important;
}
.custom-markdown-footer p {
    font-size: 18px !important;
}

.titles label {
    font-size: 15px !important;
}

.custom-radio input[type="radio"] {
    width: 30px !important;
    height: 30px !important;
}

.custom-radio label {
    font-size: 18px !important;
    display: flex;
    cursor: pointer;
    width: 30%;
    justify-content: center;
    align-items: center;
}

.custom-text label {
    font-size: 18px !important;
}

.custom-text textarea {
    font-size: 20px !important;
}
"""

with gr.Blocks(
    title="Scribble Draw",
    css=custom_css
) as demo:
    gr.Markdown("# Scribble Draw", elem_classes="custom-markdown")
    gr.Markdown(value="Draw something on the sketchpad and let the AI detect and enhance your image. Select the generation task type, choose a style and optionally a background, click 'Generate image'.", elem_classes="custom-markdown")
    with gr.Row():
        with gr.Column(
            scale=1
        ):
            sketchpad = gr.Sketchpad(width=800, height=660, canvas_size=(768, 512), type='numpy', layers=False, brush=gr.Brush(default_size=5, colors=["#000000"], color_mode="fixed"), label="Sketchpad", elem_classes=["sketchpad", "titles"])
            radio = gr.Radio(["Detect object class and generate", "Describe the scene with VLM and generate", "Generate based on edges without semantic info"], value="Detect object class and generate", interactive=True, 
                             label = "Generation task type", elem_classes="custom-radio")
            with gr.Row():
                style = gr.Dropdown(choices=["Realism", "Photorealistic CGI", "Impressionism", "Surrealism", "Pop Art", "Pixel Art", "Sketch & Ink Art", "Futurism", "Gothic", "Minimalism", "Anime"], label="Style", value="Realism", allow_custom_value=False, elem_classes="custom-dropdown")
                background = gr.Dropdown(choices=["None", "Natural", "Urban", "Studio Lighting", "Fantasy"], label="Background", value="None", allow_custom_value=False, elem_classes="custom-dropdown")
            
            btn = gr.Button(value="Generate image", elem_classes="submit-button")

        with gr.Column(
            scale=1
        ):
            out = gr.Image( width=800, height=660, show_download_button=False, label="Generated image", elem_classes="titles")
            info_msg = gr.Textbox(label="Info", interactive=False, elem_classes="custom-text")

            gr.Markdown("# Contact Information", elem_classes="custom-markdown-footer")
            with gr.Row(elem_classes="custom-footer"):
                with gr.Column():
                    gr.Markdown("**Author:** Rainer Kõiv", elem_classes="custom-markdown-footer")
                    gr.Markdown("**Email:** `rainer.k6iv@gmail.com`", elem_classes="custom-markdown-footer")
                with gr.Column():
                    gr.Markdown("**Supervisor:** Ardi Tampuu, PhD", elem_classes="custom-markdown-footer")
                    gr.Markdown("**Email:** `ardi.tampuu@ut.ee`", elem_classes="custom-markdown-footer")
   
    btn.click(fn=enhance_image, inputs=[sketchpad, radio, style, background], outputs=[out, info_msg])
    
demo.launch()