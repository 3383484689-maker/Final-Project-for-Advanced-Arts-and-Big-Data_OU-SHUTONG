%%writefile app.py
import streamlit as st
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from diffusers import DPMSolverMultistepScheduler
import torch
from PIL import Image

# ----------------------------------------
# Load Model
# ----------------------------------------

BASE_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE"
CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_openpose"

@st.cache_resource
def load_pipeline():
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL,
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    return pipe

pipe = load_pipeline()

# ----------------------------------------
# Style Dictionary (10 styles)
# ----------------------------------------

STYLE_PROMPTS = {
    "Forest Fantasy": """
    forest fantasy portrait, soft green ambient lighting, magical atmosphere,
    moss, leaves, fantasy mood, ethereal aesthetic, fairy-like elements,
    4k high-quality photography
    """,

    "Studio Portrait": """
    studio portrait photography, professional softbox lighting,
    magazine cover aesthetic, clean background, high-end beauty photograph
    """,

    "Film Look": """
    35mm film portrait, soft light, warm tones, cinematic grain, Kodak Portra style,
    vintage movie feeling, dreamy highlights
    """,

    "Cyberpunk": """
    cyberpunk portrait, neon magenta and blue lights, futuristic city reflections,
    holographic glow, sci-fi atmosphere, cinematic cyber world
    """,

    "Classical Oil Painting": """
    classical oil painting portrait, renaissance lighting, Rembrandt style,
    dramatic shadows, textured oil brush strokes, golden tones
    """,

    "Ghibli": """
    Studio Ghibli style portrait, soft pastel tones, gentle sunlight, warm mood,
    hand-painted texture, cute and whimsical
    """,

    "Watercolor": """
    watercolor portrait, transparent watercolor wash, soft brush strokes,
    hand-painted aesthetic, light bleeding colors, elegant illustration
    """,

    "K-style": """
    Korean studio portrait, soft natural lighting, pastel background, smooth skin tone,
    minimalist and clean photography style
    """,

    "Portrait Realism": """
    hyper-realistic portrait, natural skin texture, accurate lighting,
    crisp details, premium studio realism
    """,

    "Fantasy Magic": """
    fantasy magic portrait, glowing particles, magical light, ethereal atmosphere,
    fantasy cinematic visuals, enchanted aura
    """
}

# ----------------------------------------
# Streamlit UI
# ----------------------------------------

st.set_page_config(page_title="AI Artistic Portrait Generator", layout="centered")

st.title("🎨 AI Artistic Portrait Generator")
st.write("Upload a selfie → choose a style → get your artistic portrait!")

uploaded_img = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

selected_style = st.selectbox("Choose a style", list(STYLE_PROMPTS.keys()))

extra_prompt = st.text_area(
    "Optional: Add extra prompt",
    "cinematic lighting, sharp details, high quality"
)

if uploaded_img and st.button("Generate"):
    with st.spinner("Generating artistic portrait... Please wait..."):
        input_image = Image.open(uploaded_img).convert("RGB")

        full_prompt = STYLE_PROMPTS[selected_style] + ", " + extra_prompt

        result = pipe(
            prompt=full_prompt,
            image=input_image,
            num_inference_steps=30,
        ).images[0]

        st.image(result, caption=f"{selected_style} Style Output", use_column_width=True)

        # Option to download
        result.save("output.png")
        with open("output.png", "rb") as f:
            st.download_button(
                "Download Output Image",
                f,
                file_name="artistic_portrait.png",
                mime="image/png"
            )
