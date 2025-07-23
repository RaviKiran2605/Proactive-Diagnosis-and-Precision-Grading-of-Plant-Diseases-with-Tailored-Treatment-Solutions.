import streamlit as st
import os

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def display_welcome_page():
    # Load external CSS
    css_file = os.path.join(os.path.dirname(__file__), "../styles/wel.css")
    load_css(css_file)

    # Centered page title
    st.markdown('<h1 class="center-title">🌱 Plant Disease Detection System</h1>', unsafe_allow_html=True)

    # Add a smaller banner image
    st.image(
        "styles/green_background.jpg", 
        caption="Effortlessly identify plant diseases with AI",
        use_container_width=False
    )

    # Introduction
    st.markdown(
        """
        <p>Welcome to the Plant Disease Detection System! Our platform uses advanced 
        machine learning models to accurately identify diseases affecting your plants 
        based on leaf images.</p>

        <p>### 🌟 Key Features:</p>
        <ul>
            <li><strong>Disease Prediction:</strong> Upload leaf images for detailed analysis.</li>
            <li><strong>Cure Suggestions:</strong> Get actionable steps to mitigate diseases.</li>
            <li><strong>Insights Dashboard:</strong> Visualize trends and statistics related to plant health.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Check the Plant Disease Here"):
            st.session_state.page = "Prediction"  # Change page to Prediction
    with col2:
        if st.button("Go to Dashboard"):
            st.session_state.page = "Dashboard"  # Change page to Dashboard

# Ensure to call the display_welcome_page function to render the page
if __name__ == "__main__":
    display_welcome_page()
