import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline

# 🖌 Add custom styles
def add_custom_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)),
            url('https://images.unsplash.com/photo-1519389950473-47ba0277781c');
            background-size: cover;
            background-attachment: fixed;
            font-family: 'Segoe UI', sans-serif;
        }

        .glass-box {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 35px;
            margin: 40px auto;
            max-width: 850px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        /* Wrap actual uploader input */
        .upload-box {
            border: 2px dashed #ccc;
            padding: 40px;
            border-radius: 12px;
            background-color: rgba(255, 255, 255, 0.05);
            transition: all 0.3s ease-in-out;
            box-shadow: inset 0 0 0 transparent;
        }

        .upload-box:hover {
            border-color: #22d4fd;
            box-shadow: 0 0 12px #22d4fd, 0 0 20px #5be0ff;
            animation: pulse 1s infinite alternate;
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 5px #22d4fd; }
            100% { box-shadow: 0 0 15px #22d4fd, 0 0 25px #66f; }
        }

        .arrow-down {
            text-align: center;
            font-size: 2rem;
            animation: bounce 1s infinite;
            color: #aaa;
        }

        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(8px); }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_custom_styles()

# 💠 Main Glass Container
st.markdown('<div class="glass-box">', unsafe_allow_html=True)

# 🧠 Heading
st.markdown("<h1 style='text-align:center; color:white;'>AI Certificate Verifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#eee;'>Upload a certificate PDF and let AI verify its authenticity</p>", unsafe_allow_html=True)
st.markdown('<div class="arrow-down">⬇</div>', unsafe_allow_html=True)
st.markdown("---")

# 📦 Upload Box (Styled with real uploader inside)
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drag & Drop or Click to Browse", type=["pdf"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# 🔁 Load model (cached)
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-small")

generator = load_model()

# ✅ Handle Uploaded PDF
if uploaded_file:
    try:
        pdf = PdfReader(uploaded_file)
        text = "".join([page.extract_text() for page in pdf.pages if page.extract_text()])

        st.markdown("### 📝 Extracted Certificate Text")
        st.text_area("Extracted Content", text, height=250)

        st.markdown("### 🤖 AI Authenticity Verdict")
        with st.spinner("Analyzing with our AI model..."):
            prompt = f"Is this certificate authentic? Here's the content: {text}"
            result = generator(prompt, max_length=100)[0]["generated_text"]
            st.success(result)

    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")

# 🔚 Close Glass Box
st.markdown('</div>', unsafe_allow_html=True)
