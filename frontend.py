import backend as backend
import gradio as gr
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

def enhance_image(sketchpad, style, lang):
    canvas_size = (768, 512) # 768x512 pixel canvas size
    radio = "Describe the scene with VLM and generate"
    if style == "Futurism":
        background = "Urban"
    else:
        background = "Natural"
    enhanced_image, message = backend.enhance_drawing(sketchpad, radio, style, background, canvas_size, lang) # enhance_drawing function from backend.py
    return enhanced_image, message 

def get_translations(lang):
    return {
        "en": {
            "title": "# Scribble Draw",
            "description": "Scribble something on the sketchpad. AI will try to understand and enhance it.",
            "sketchpad": "Sketchpad",
            "realism": "Realism",
            "gothic": "Gothic",
            "pop_art": "Pop Art",
            "fantasy": "Fantasy",
            "futurism": "Futurism",
            "sketch": "Sketch & Ink Art",
            "impressionism": "Impressionism",
            "anime": "Anime",
            "generate": "Enhance image with AI",
            "generated_image": "Enhanced image",
            "info": "Info",
            "read_more": "Read more:",
            "qr_code": "static/qr-en.png",
            "logo": "static/arvutiteaduse_instituut_eng_blue_long.svg"
        },
        "et": {
            "title": "# Scribble Draw",
            "description": "Kritselda midagi joonistusalale. Tehisintellekt proovib seda mõista ja täiendada.",
            "sketchpad": "Joonistusala",
            "realism": "Realism",
            "gothic": "Gooti stiil",
            "pop_art": "Popkunst",
            "fantasy": "Fantaasia",
            "futurism": "Futurism",
            "sketch": "Joonistus",
            "impressionism": "Impressionism",
            "anime": "Anime",
            "generate": "Täienda pilti AI abiga",
            "generated_image": "Täiendatud pilt",
            "info": "Info",
            "read_more": "Loe rohkem selle töö kohta:",
            "qr_code": "static/qr-et.png",
            "logo": "static/arvutiteaduse_instituut_est_sinine.svg"
        }
    }[lang]

def switch_language(lang):
    t = get_translations(lang)
    return (t["title"], t["description"], gr.update(label=t["sketchpad"]), 
            t["realism"], t["gothic"], t["pop_art"], t["fantasy"], t["futurism"], t["sketch"], t["impressionism"], t["anime"], 
            t["generate"], gr.update(label=t["generated_image"]), gr.update(label=t["info"]), t["read_more"],
            gr.update(value=t["qr_code"]), gr.update(value=t["logo"]), lang)

# For color change
def set_language(lang):
    return switch_language(lang) + (
        gr.update(elem_classes=["lang-btn", "selected-lang" if lang == "en" else "lang-btn"]),
        gr.update(elem_classes=["lang-btn", "selected-lang" if lang == "et" else "lang-btn"]),
    )
# For style button outline color change
def set_style(style):
    def class_for(s):
        return ["style-btn", "selected-style"] if s == style else ["style-btn"]
    
    return (
        style,  # update internal style state
        gr.update(elem_classes=class_for("Realism")),
        gr.update(elem_classes=class_for("Gothic")),
        gr.update(elem_classes=class_for("Pop Art")),
        gr.update(elem_classes=class_for("Fantasy")),
        gr.update(elem_classes=class_for("Futurism")),
        gr.update(elem_classes=class_for("Sketch & Ink Art")),
        gr.update(elem_classes=class_for("Impressionism")),
        gr.update(elem_classes=class_for("Anime")),
    )

custom_css = """
/* Toolbar buttons */
button[title="View in full screen"],
button[title="Exit full screen"],
button[title="Clear canvas"], 
button[title="Undo"], 
button[title="Redo"],
button[title="Erase button"],
button[title="Draw button"] {
    background-color: #2C5697 !important;
    width: 40px !important;
    height: 40px !important;
    font-size: 20px !important;
    color: white !important;
}

button[title="Transform button"] {
    display: none !important;
}

button[title="Draw button"] {
    margin-right: 15px !important;
}

footer {visibility: hidden;}

.submit-button {
    background-color: #2C5697 !important;
    color: white !important;
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


/* AHHAA simplified additions */
.style-btn {
    background-size: cover;
    width: 180px;
    height: 120px;
    border: none;
    outline: none;
    cursor: pointer;
    color: white;
    display: flex;
    align-items: flex-end;
    justify-content: center;
    text-shadow: 
        -1px -1px 0 #000, 
         1px -1px 0 #000, 
        -1px  1px 0 #000, 
         1px  1px 0 #000;
}

#realism-btn {
    background-image: url('/static/realism.png?v=2');
}
#gothic-btn {
    background-image: url('/static/gothic.png?v=2');
}
#pop_art-btn {
    background-image: url('/static/pop_art.png?v=2');
}
#fantasy-btn {
    background-image: url('/static/fantasy.png?v=2');
}
#futurism-btn {
    background-image: url('/static/futurism.png?v=2');
}
#sketch-btn {
    background-image: url('/static/sketch.png?v=2');
}
#impressionism-btn {
    background-image: url('/static/impressionism.png?v=2');
}
#anime-btn {
    background-image: url('/static/anime.png?v=2');
}

.selected-style {
    outline: 5px solid #2C5697 !important;
    box-shadow: inset 0 0 0 1px white; /* Inner white border */
}
.logo {
    padding: 25px;
}
.lang-btn {
    background-color: #f0f0f0;
    color: #000;
    font-weight: bold;
    padding: 8px 16px;
    font-size: 16px;
    transition: background-color 0.3s ease;
}

.lang-btn:hover {
    background-color: #d9e2f3;
}

.lang-btn.selected-lang {
    background-color: #2C5697;
    color: white;
}
"""

def create_gradio_app():
    with gr.Blocks(
        title="Scribble Draw",
        css=custom_css
    ) as demo:
        language_state = gr.State(value="en")  # default: English
        style_state = gr.State(value="Realism")  # Default style

        with gr.Row():
            with gr.Column():
                title_markdown = gr.Markdown(get_translations("en")["title"], elem_classes="custom-markdown")
            with gr.Column():
                with gr.Row():
                    lang_en = gr.Button("🇬🇧 English", elem_classes=["lang-btn", "selected-lang"])
                    lang_et = gr.Button("🇪🇪 Eesti", elem_classes="lang-btn")

        desc_markdown = gr.Markdown(get_translations("en")["description"], elem_classes="custom-markdown")
        with gr.Row():
            with gr.Column(
                scale=1
            ):
                sketchpad = gr.Sketchpad(width=800, height=660, canvas_size=(768, 512), type='numpy', layers=False, brush=gr.Brush(default_size=5, colors=["#000000"], color_mode="fixed"), label=get_translations("en")["sketchpad"], elem_classes=["sketchpad", "titles"])
                #radio = gr.Radio(["Detect object class and generate", "Describe the scene with VLM and generate", "Generate based on edges without semantic info"], value="Detect object class and generate", interactive=True, 
                #                 label = "Generation task type", elem_classes="custom-radio")
                with gr.Row():
                    #style = gr.Dropdown(choices=["Realism", "Photorealistic CGI", "Impressionism", "Surrealism", "Pop Art", "Pixel Art", "Sketch & Ink Art", "Futurism", "Gothic", "Minimalism", "Anime", "Fantasy"], label="Style", value="Realism", allow_custom_value=False, elem_classes="custom-dropdown")
                    #background = gr.Dropdown(choices=["None", "Natural", "Urban", "Studio Lighting", "Fantasy"], label="Background", value="None", allow_custom_value=False, elem_classes="custom-dropdown")
                    realism_btn = gr.Button(get_translations("en")["realism"], elem_id="realism-btn", elem_classes="style-btn")
                    gothic_btn = gr.Button(get_translations("en")["gothic"], elem_id="gothic-btn", elem_classes="style-btn")
                    pop_art_btn = gr.Button(get_translations("en")["pop_art"], elem_id="pop_art-btn", elem_classes="style-btn")
                    fantasy_btn = gr.Button(get_translations("en")["fantasy"], elem_id="fantasy-btn", elem_classes="style-btn")
                with gr.Row():
                    futurism_btn = gr.Button(get_translations("en")["futurism"], elem_id="futurism-btn", elem_classes="style-btn")
                    sketch_btn = gr.Button(get_translations("en")["sketch"], elem_id="sketch-btn", elem_classes="style-btn")
                    impressionism_btn = gr.Button(get_translations("en")["impressionism"], elem_id="impressionism-btn", elem_classes="style-btn")
                    anime_btn = gr.Button(get_translations("en")["anime"], elem_id="anime-btn", elem_classes="style-btn")
                
                realism_btn.click(
                    fn=lambda: set_style("Realism"),
                    inputs=[],
                    outputs=[
                        style_state,
                        realism_btn, gothic_btn, pop_art_btn, fantasy_btn,
                        futurism_btn, sketch_btn, impressionism_btn, anime_btn
                    ]
                )
                gothic_btn.click(
                    fn=lambda: set_style("Gothic"),
                    inputs=[],
                    outputs=[
                        style_state,
                        realism_btn, gothic_btn, pop_art_btn, fantasy_btn,
                        futurism_btn, sketch_btn, impressionism_btn, anime_btn
                    ]
                )
                pop_art_btn.click(
                    fn=lambda: set_style("Pop Art"),
                    inputs=[],
                    outputs=[
                        style_state,
                        realism_btn, gothic_btn, pop_art_btn, fantasy_btn,
                        futurism_btn, sketch_btn, impressionism_btn, anime_btn
                    ]
                )
                fantasy_btn.click(
                    fn=lambda: set_style("Fantasy"),
                    inputs=[],
                    outputs=[
                        style_state,
                        realism_btn, gothic_btn, pop_art_btn, fantasy_btn,
                        futurism_btn, sketch_btn, impressionism_btn, anime_btn
                    ]
                )
                futurism_btn.click(
                    fn=lambda: set_style("Futurism"),
                    inputs=[],
                    outputs=[
                        style_state,
                        realism_btn, gothic_btn, pop_art_btn, fantasy_btn,
                        futurism_btn, sketch_btn, impressionism_btn, anime_btn
                    ]
                )
                sketch_btn.click(
                    fn=lambda: set_style("Sketch & Ink Art"),
                    inputs=[],
                    outputs=[
                        style_state,
                        realism_btn, gothic_btn, pop_art_btn, fantasy_btn,
                        futurism_btn, sketch_btn, impressionism_btn, anime_btn
                    ]
                )
                impressionism_btn.click(
                    fn=lambda: set_style("Impressionism"),
                    inputs=[],
                    outputs=[
                        style_state,
                        realism_btn, gothic_btn, pop_art_btn, fantasy_btn,
                        futurism_btn, sketch_btn, impressionism_btn, anime_btn
                    ]
                )
                anime_btn.click(
                    fn=lambda: set_style("Anime"),
                    inputs=[],
                    outputs=[
                        style_state,
                        realism_btn, gothic_btn, pop_art_btn, fantasy_btn,
                        futurism_btn, sketch_btn, impressionism_btn, anime_btn
                    ]
                )
                
                btn = gr.Button(value=get_translations("en")["generate"], elem_classes="submit-button")

            with gr.Column(
                scale=1
            ):
                out = gr.Image(width=800, height=660, show_download_button=False, show_fullscreen_button=False, label=get_translations("en")["generated_image"], elem_classes="titles")
                info_msg = gr.Textbox(label=get_translations("en")["info"], interactive=False, elem_classes="custom-text")

                #gr.Markdown("# Contact Information", elem_classes="custom-markdown-footer")
                #with gr.Row(elem_classes="custom-footer"):
                #    with gr.Column():
                #        gr.Markdown("**Author:** Rainer Kõiv", elem_classes="custom-markdown-footer")
                #        gr.Markdown("**Email:** `rainer.k6iv@gmail.com`", elem_classes="custom-markdown-footer")
                #    with gr.Column():
                #        gr.Markdown("**Supervisor:** Ardi Tampuu, PhD", elem_classes="custom-markdown-footer")
                #        gr.Markdown("**Email:** `ardi.tampuu@ut.ee`", elem_classes="custom-markdown-footer")
                read_more_md = gr.Markdown(get_translations("en")["read_more"], elem_classes="custom-markdown")
                with gr.Row():
                    qr_code = gr.Image(value=get_translations("en")["qr_code"], type="filepath", format="png", height=120, width=120, interactive=False, show_download_button=False, show_fullscreen_button=False, show_label=False)
                    logo = gr.Image(value=get_translations("en")["logo"], type="filepath", format="svg", height=120, width=260, interactive=False, show_download_button=False, show_fullscreen_button=False, show_label=False, elem_classes="logo")
        
        btn.click(fn=enhance_image, inputs=[sketchpad, style_state, language_state], outputs=[out, info_msg])
        
        #lang_en.click(fn=lambda: switch_language("en"), inputs=[], outputs=[
        #    title_markdown, desc_markdown, sketchpad, realism_btn, gothic_btn, pop_art_btn, fantasy_btn, futurism_btn, sketch_btn, impressionism_btn, anime_btn, btn, out, info_msg, read_more_md, qr_code, logo, language_state
        #], show_progress=False, preprocess=True, postprocess=True)
        lang_en.click(fn=lambda: set_language("en"), inputs=[], outputs=[
            title_markdown, desc_markdown, sketchpad, realism_btn, gothic_btn, pop_art_btn, fantasy_btn, futurism_btn, sketch_btn, impressionism_btn, anime_btn,
            btn, out, info_msg, read_more_md, qr_code, logo, language_state,
            lang_en, lang_et
        ], show_progress=False, preprocess=True, postprocess=True)
        lang_et.click(fn=lambda: set_language("et"), inputs=[], outputs=[
            title_markdown, desc_markdown, sketchpad, realism_btn, gothic_btn, pop_art_btn, fantasy_btn, futurism_btn, sketch_btn, impressionism_btn, anime_btn,
            btn, out, info_msg, read_more_md, qr_code, logo, language_state,
            lang_en, lang_et
        ], show_progress=False, preprocess=True, postprocess=True)
        #lang_et.click(fn=lambda: switch_language("et"), inputs=[], outputs=[
        #    title_markdown, desc_markdown, sketchpad, realism_btn, gothic_btn, pop_art_btn, fantasy_btn, futurism_btn, sketch_btn, impressionism_btn, anime_btn, btn, out, info_msg, read_more_md, qr_code, logo, language_state
        #], show_progress=False, preprocess=True, postprocess=True)
    
        gr.HTML("""
        <div id="inject-script"></div>
        <script>
        (function() {
            let s = document.createElement("script");
            s.innerHTML = `
                function setupButtons() {
                    const styleButtons = document.querySelectorAll('.style-btn');
                    styleButtons.forEach(button => {
                        button.onclick = () => {
                            styleButtons.forEach(btn => btn.classList.remove('selected-style'));
                            button.classList.add('selected-style');
                        };
                    });

                const langButtons = document.querySelectorAll('.lang-btn');
                langButtons.forEach(button => {
                    button.onclick = () => {
                        langButtons.forEach(btn => btn.classList.remove('selected-lang'));
                        button.classList.add('selected-lang');
                        };
                    });
                }

                // Setup initially
                setupButtons();

                // Watch for UI updates
                const observer = new MutationObserver(setupButtons);
                observer.observe(document.body, { childList: true, subtree: true });
            `;
            document.getElementById("inject-script").appendChild(s);
        })();
        </script>
        """)

    return demo
    #demo.launch()
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

gradio_app = create_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/")