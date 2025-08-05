import streamlit as st

def elegant_sidebar():
    with st.sidebar:
        # Logo
        st.image("assets/images/logo.png", width=60)
        # App title with gradient font color (heading only)
        st.markdown(
            """
            <span style="
                font-size:1.65rem;
                font-weight:900;
                background: linear-gradient(315deg, #f5f5f5 0%, #e34234 74%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-fill-color: transparent;
                display:inline-block;
            ">
                BoinkVision AI
            </span>
            """,
            unsafe_allow_html=True,
        )
        st.write("")  # Spacer

        st.markdown("**Real-time ISS asset detection**")
        st.caption(
            "Detects Fire Extinguishers, Oxygen Tanks, Toolboxes instantly. "
            "[Explore Falcon platform →](https://falcon.duality.ai/)"
        )

        st.markdown("---")
        st.caption("© 2025 OINK BOINK | Duality AI Space Station Hackathon")
