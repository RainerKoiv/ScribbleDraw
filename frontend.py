import backend as backend
import gradio as gr
import numpy as np

def enhance_image(sketchpad, radio, style, background):
    enhanced_image, message = backend.enhance_drawing(sketchpad, radio, style, background) # kutsub backendist pilditäiendusfunktsiooni
    return enhanced_image, message 


with gr.Blocks() as demo:
    gr.Markdown("# Scribble Draw")
    gr.Markdown("Draw a picture and choose a style to enhance it.")
    with gr.Row():
        with gr.Column():
            sketchpad = gr.Sketchpad(canvas_size=(512,512), type='numpy', layers=False, brush=gr.Brush(default_size=5, colors=["#000000"], color_mode="fixed"))
            radio = gr.Radio(["Specific object", "A specific scene","A random scene"], value="Specific object", interactive=True, label = "Choose whether you want to improve one specific object, a scene with multiple objects or to create a random scene from your drawing:")
            style = gr.Dropdown(choices=["Realism", "Photorealistic CGI", "Impressionism", "Surrealism", "Pop Art", "Pixel Art", "Sketch & Ink Art", "Futurism", "Gothic", "Minimalism", "Anime"], label="Style", value="Realism", allow_custom_value=False)
            background = gr.Dropdown(choices=["None", "Natural", "Urban", "Studio Lighting", "Fantasy"], label="Background", value="None", allow_custom_value=False)
        with gr.Column():
            out = gr.Image(show_download_button=False)
            info_msg = gr.Textbox(label="Info", interactive=False)
    btn = gr.Button("Submit")
    btn.click(fn=enhance_image, inputs=[sketchpad, radio, style, background], outputs=[out, info_msg])

demo.launch()


#gr.Interface(
#    fn=enhance_image,
#    inputs=gr.Sketchpad(canvas_size=(512,512), type='numpy', layers=False, brush=gr.Brush(default_size="auto", colors=["#000000"], color_mode="fixed")),
#    style=gr.Dropdown(["natural background", "anime style", "realistic"], label="Style"),
#    outputs="image",
#    #outputs="textbox", testi objekti tuvastust
#    title="Scribble Draw"
#).launch()
