import streamlit as st
import requests
import io
import zipfile
import os
import warnings


warnings.filterwarnings("ignore")


try:
    from requests.packages import urllib3
    urllib3.disable_warnings()
except ImportError:
    pass


st.set_page_config(page_title="Metal Surface Defect Detection", layout="wide")


st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
        color: blue;  /* Set text color to blue */
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown('<h1 class="centered-title">Metal Surface Defect Detection</h1>', unsafe_allow_html=True)
st.markdown("Upload an image or a folder to detect defects on metal surfaces using machine learning.")


url = 'http://127.0.0.1:5000/Metal_surface_pred'
#folder_url = 'http://127.0.0.1:5000/Metal_surface_pred_folder'
test_url = 'http://127.0.1:5000/'


def zip_folder_in_memory(folder_path):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=folder_path)
                zip_file.write(file_path, arcname)
    zip_buffer.seek(0)
    return zip_buffer


st.subheader("Upload Images")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        analyze_button = st.button("Analyze Image")


with st.container():
    col1, col2, col3 = st.columns([1, 2, 1]) 
    if uploaded_image is not None and analyze_button:
        with st.spinner("Analyzing..."):
            files = {'image': uploaded_image}
            response = requests.post(url, files=files)
            
            if response.status_code == 200:
                with col1:
                    st.success("Request successful!")
            else:
                with col1:
                    st.error(f"Request failed with status code {response.status_code}")


if uploaded_image is not None:
    with st.container():
        col1, col2, col3 = st.columns([2, 1, 2]) 
        with col1:
            st.image(uploaded_image, caption="Uploaded Image", width=800, use_container_width=True)
        
        if analyze_button and response.status_code == 200:
            with col3:
                st.image(response.content, caption="Processed Image", width=800, use_container_width=True)


st.markdown("---")
st.markdown("Made by Sai Ethihas Chanda | currently hosted at http://127.0.0.1:5000")