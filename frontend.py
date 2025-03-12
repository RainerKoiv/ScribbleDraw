import backend as backend
import gradio as gr
import numpy as np

def enhance_image(sketchpad, radio, style, background):
    canvas_size = (768, 512) # 768x512 pikslit
    enhanced_image, message = backend.enhance_drawing(sketchpad, radio, style, background, canvas_size) # kutsub backendist pilditäiendusfunktsiooni
    return enhanced_image, message 


custom_css = """
/* Increase the size of the toolbar buttons in Sketchpad */
button[title="View in full screen"],
button[title="Clear canvas"], 
button[title="Undo"], 
button[title="Redo"],
button[title="Erase button"],
button[title="Draw button"] {
    width: 40px !important;
    height: 40px !important;
    font-size: 18px !important;
}

/* Hide the Transform button in the Sketchpad */
button[title="Transform button"] {
    display: none !important;
}
"""

with gr.Blocks(
    title="Scribble Draw",
    css=custom_css
) as demo:
    gr.Markdown("# Welcome to Scribble Draw!")
    gr.Markdown("Here you can draw anything you want on the sketchpad. Select the generation task type, choose a style and optionally a background, and let the AI enhance your drawing by clicking 'Submit'.")
    with gr.Row():
        with gr.Column(
            scale=2
        ):
            sketchpad = gr.Sketchpad(width=800, height=640, canvas_size=(768, 512), type='numpy', layers=False, brush=gr.Brush(default_size=5, colors=["#000000"], color_mode="fixed"), label="Sketchpad")
            radio = gr.Radio(["Enhance one object", "Enhance multiple objects", "Generate a random scene from your drawing"], value="Enhance one object", interactive=True, 
                             label = "Generation task type. First option works best with a single object - object detection will be used. Second option works best with multiple objects - a caption will be generated.")
            style = gr.Dropdown(choices=["Realism", "Photorealistic CGI", "Impressionism", "Surrealism", "Pop Art", "Pixel Art", "Sketch & Ink Art", "Futurism", "Gothic", "Minimalism", "Anime"], label="Style", value="Realism", allow_custom_value=False)
            background = gr.Dropdown(choices=["None", "Natural", "Urban", "Studio Lighting", "Fantasy"], label="Background", value="None", allow_custom_value=False)
        with gr.Column(
            scale=2
        ):
            out = gr.Image( width=800, height=640, show_download_button=False, label="Image")
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
