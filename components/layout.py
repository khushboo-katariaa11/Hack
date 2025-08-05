import streamlit as st
import base64

def inject_landing_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
        body, .main, .block-container {
            background: #181C21 !important;
            font-family: 'Inter', sans-serif !important;
        }
        .landing-hero {
            min-height: 90vh;
            padding-top: 4vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .landing-logo {
            width: 86px;
            margin-bottom: 2rem;
            border-radius: 13px;
            box-shadow: 0 1px 14px #e3423475, 0 0 15px #fff1;
            object-fit: contain;
            background: #232333;
        }
        .gradient-heading-hero {
            font-size: 2.4rem;
            font-weight: 800;
            background-image: linear-gradient(315deg, #f5f5f5 0%, #e34234 74%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.4em;
            letter-spacing: 0.01em;
            text-align: center;
            line-height: 1.18;
        }
        .subtitle-hero {
            margin: 0 0 1.8em 0;
            font-weight: 700;
            font-size: 1.45rem;
            color: #e0e0e0;
            letter-spacing: 0.012em;
            text-align: center;
        }
        .landing-subheader {
            max-width: 580px;
            margin: 0 auto;
            text-align: left; /* Changed from center to left */
            color: #CCCCCF;
            font-size: 1.09rem;
            line-height: 1.67;
            padding: 1.1em 0 2em 0;
            margin-bottom: 0.2em;
        }
        .stButton > button {
            /* border: 2px solid #e34234; */ /* Removed border */
            background-color: transparent;
            color: #FFFFFF;
            padding: 13px 36px;
            width: 220px;
            font-size: 1.13rem;
            font-weight: 600;
            border-radius: 9px;
            transition: all 0.23s;
            margin-top: 2em;
        }
        .stButton > button:hover {
            background-color: #e34234;
            /* border-color: #e34234; */ /* Removed border on hover */
            color: #fff;
            transform: translateY(-2px) scale(1.035);
            box-shadow: 0 4px 20px #e3423466, 0 4px 25px #fff1;
        }
        </style>
        """, unsafe_allow_html=True
    )

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def elegant_landing():
    inject_landing_css()
    

    st.markdown(
        "<span class='gradient-heading-hero'>BoinkVision: ISS Safety Asset Monitor</span>",
        unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle-hero'  style='text-align:left;'>Space-grade AI for Real-Time Critical Equipment Detection</div>",
        unsafe_allow_html=True)
    st.markdown("""
        <p class='landing-subheader'>
            Our mission: ensuring safety and automation for the next generation of space stations—by blending best-in-class computer vision with robust synthetic data from Duality AI's Falcon platform.<br>
            <span style='color:#e34234;'><b>Start monitoring, discover insights, stay compliant—faster than ever before.</b></span>
        </p>
        </div>
    """, unsafe_allow_html=True)

    return st.button("Get Started", key="go_app")

def page_heading(text):
    st.markdown(
        f"<h2 style=\"margin-top:1rem;font-family:'Inter',sans-serif;font-size:1.55rem;font-weight:700;background-image:linear-gradient(315deg,#f5f5f5 0%,#e34234 74%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;\">{text}</h2>",
        unsafe_allow_html=True)

def elegant_section(title, description=""):
    st.markdown(f"""
    <div class='elegant-card'>
        <h3 style="font-family:'Inter',sans-serif;font-size:1.08rem;font-weight:bold;color:#e34234;">{title}</h3>
        <p style="color:#A5A7AE;font-size: 1rem;line-height:1.5;">{description}</p>
    </div>
    """, unsafe_allow_html=True)

def section_close():
    pass
