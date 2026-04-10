import streamlit as st
import requests
import io
import sys
import zipfile
import os
import warnings
import tempfile
import time
import base64
import pandas as pd
import plotly.express as px
import threading

# Set environment variables for headless mode BEFORE any cv2 imports
os.environ['HEADLESS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['DISPLAY'] = ''

from PIL import Image

# LAZY LOAD YOLO - only import when needed to avoid OpenGL errors
_yolo_model = None
def get_yolo_model(model_path="best_neu.pt"):
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(model_path)
    return _yolo_model

# Suppress warnings
# warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
        color: blue;
    }
    .sidebar-footer {
        margin-top: 400px;
        font-size: 14px;
        font-style: italic;
        color: #555555;
        line-height: 1.5;
        padding: 10px;
    }
    .hardcoded-table {
        width: 50%;  /* Adjust table width */
        margin: 0 auto;  /* Center the table */
        border-collapse: collapse;
        font-size: 14px;
    }
    .hardcoded-table th, .hardcoded-table td {
        border: 1px solid #dddddd;
        padding: 8px;
        text-align: center;
    }
    .hardcoded-table th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)   

    
# ============================================
# START FLASK BACKEND IN BACKGROUND THREAD
# ============================================
def start_flask_server():
    """Start Flask server in a background thread"""
    try:
        # Import Flask app from flask_script (same directory)
        app_dir = os.path.dirname(os.path.abspath(__file__))
        if app_dir not in sys.path:
            sys.path.insert(0, app_dir)
        from flask_script import app as flask_app
        
        # Run Flask app with threading disabled to avoid conflicts
        flask_app.run(debug=False, threaded=True, use_reloader=False, port=5000, host='127.0.0.1')
    except Exception as e:
        print(f"Error starting Flask server: {e}")

# Start Flask in a background thread only once
if 'flask_started' not in st.session_state:
    flask_thread = threading.Thread(target=start_flask_server, daemon=True)
    flask_thread.start()
    st.session_state.flask_started = True
    time.sleep(2)  # Give Flask time to start

# API endpoints
url = 'http://127.0.0.1:5000/Metal_surface_pred'
test_url = 'http://127.0.0.1:5000/'
normal_url = 'http://127.0.0.1:5000/normal'

# Initialize session state for results
if 'results' not in st.session_state:
    st.session_state.results = []

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

def handle_model_selection(model):
    """Function to handle dropdown selection"""
    st.session_state.selected_model = model
    st.write(f"Selected model: {model}")


def safe_unlink(file_path):
    """Attempt to delete a file with retries on PermissionError."""
    if os.path.exists(file_path):
        os.unlink(file_path)


# Collapsible sidebar with tabs
with st.sidebar:
    st.header("Navigation")
    tab = st.radio("Select Tab", ["Welcome", "Single Image Detection", "Multiple Image Detection", "Results"])
    st.markdown(
        """
        <div class="sidebar-footer">
            Developed by Sai Ethihas Chanda<br>
            Guide: Dr.Brindha G R<br>
            Contact: saiethihaschanda@gmail.com<br>
            Powered by Streamlit & YOLO
        </div>
        """,
        unsafe_allow_html=True
    )


if tab == "Welcome":
    col1, col2, col3 = st.columns([1, 2, 1])  # Middle column is wider
    with col2:
        try:
            st.image(
                r"C:\Users\saiet\OneDrive\Desktop\stream\logo.jpg",
                width=300,
                caption=None,
                clamp=True,
                output_format="auto"
            )
        except FileNotFoundError:
            st.warning("Logo image not found at 'C:\\Users\\saiet\\OneDrive\\Desktop\\stream\\logo.jpg'.")
    
    st.markdown('<h1 class="centered-title">Metal Surface Defect Detection using Fine Tuned DSL-YOLO and ML Techniques</h1>', unsafe_allow_html=True)
    st.markdown("""
    This application allows you to detect defects on metal surfaces using different modified YOLO models and compare results.
    
    **Features:**
    - **Single Image Detection**: Upload a single image to detect defects using YOLOv8n, DSL-YOLO, or a custom YOLOv8 model.
    - **Multiple Image Detection**: Upload multiple images to process them in batch and view results.
    - **Model Selection**: Choose between different detection models for tailored performance.
    
    Use the sidebar to navigate between tabs and start analyzing your images!
    """)
    st.markdown("---")
    st.markdown("Made by Sai Ethihas Chanda | currently hosted at http://127.0.0.1:5000")

# Tab 2: Single Image Detection (Original Functionality)
elif tab == "Single Image Detection":
    st.markdown('<h1 class="centered-title">Singular Image Defect Detector</h1>', unsafe_allow_html=True)
    st.markdown("Upload an image to detect defects on metal surfaces using machine learning.")

    st.subheader("Upload Image")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Dropdown box for model selection
    model_options = ["Model 1: YOLOv8n", "Model 2: DSL-YOLO", "Model 3: Custom YOLOv8"]
    selected_model = st.selectbox("Select Detection Model", model_options, on_change=handle_model_selection, args=(st.session_state.get('selected_model', model_options[0]),))

    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            analyze_button = st.button("Analyze Image")

    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1]) 
        if uploaded_image is not None and analyze_button:
            with st.spinner("Analyzing..."):
                start_time = time.time()
                result = None
                if selected_model == "Model 1: YOLOv8n":
                    try:
                        model = get_yolo_model("best_neu.pt")
                        import cv2
                        import numpy as np
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                            img = Image.open(uploaded_image)
                            img.save(tmp_file.name)
                            tmp_file_path = tmp_file.name
                        try:
                            results = model(tmp_file_path)
                            # Convert YOLO output to bytes without creating temp file
                            output_img = results[0].plot()
                            output_pil = Image.fromarray(output_img[..., ::-1])
                            # Save to bytes buffer instead of file
                            buffer = io.BytesIO()
                            output_pil.save(buffer, format="JPEG")
                            buffer.seek(0)
                            processed_image = Image.open(io.BytesIO(buffer.getvalue()))
                            with col1:
                                st.success("YOLOv8n processing successful!")
                            result = {
                                'name': uploaded_image.name,
                                'status': 'success',
                                'message': "YOLOv8n processing successful!",
                                'image': processed_image,
                                'detections': [],
                                'processing_time': time.time() - start_time
                            }
                        finally:
                            try:
                                safe_unlink(tmp_file_path)
                            except Exception as e:
                                pass  # Silently ignore
                    except Exception as e:
                        with col1:
                            st.error(f"Local YOLOv8n processing failed: {str(e)}")
                        processed_image = None
                        result = {
                            'name': uploaded_image.name,
                            'status': 'error',
                            'message': f"Local YOLOv8n processing failed: {str(e)}",
                            'image': None,
                            'detections': [],
                            'processing_time': time.time() - start_time
                        }
                elif selected_model == "Model 2: DSL-YOLO":
                    try:
                        # API call for DSL-YOLO
                        files = {'image': (uploaded_image.name, uploaded_image, uploaded_image.type)}
                        response = requests.post(normal_url, files=files)
                        
                        if response.status_code == 200:
                            with col1:
                                st.success("DSL-YOLO processing successful!")
                            processed_image = Image.open(io.BytesIO(response.content))
                            result = {
                                'name': uploaded_image.name,
                                'status': 'success',
                                'message': "DSL-YOLO processing successful!",
                                'image': processed_image,
                                'detections': [],
                                'processing_time': time.time() - start_time
                            }
                        else:
                            with col1:
                                st.error(f"Request failed with status code {response.status_code}")
                            processed_image = None
                            result = {
                                'name': uploaded_image.name,
                                'status': 'error',
                                'message': f"Request failed with status code {response.status_code}",
                                'image': None,
                                'detections': [],
                                'processing_time': time.time() - start_time
                            }
                    except Exception as e:
                        with col1:
                            st.error(f"DSL-YOLO processing failed: {str(e)}")
                        processed_image = None
                        result = {
                            'name': uploaded_image.name,
                            'status': 'error',
                            'message': f"DSL-YOLO processing failed: {str(e)}",
                            'image': None,
                            'detections': [],
                            'processing_time': time.time() - start_time
                        }
                elif selected_model == "Model 3: Custom YOLOv8":
                    try:
                        # API call for Custom YOLOv8
                        files = {'image': (uploaded_image.name, uploaded_image, uploaded_image.type)}
                        response = requests.post(url, files=files)
                        
                        if response.status_code == 200:
                            response_data = response.json()
                            if 'error' in response_data:
                                with col1:
                                    st.error(f"API error: {response_data['error']}")
                                processed_image = None
                                result = {
                                    'name': uploaded_image.name,
                                    'status': 'error',
                                    'message': f"API error: {response_data['error']}",
                                    'image': None,
                                    'detections': [],
                                    'processing_time': time.time() - start_time
                                }
                            else:
                                with col1:
                                    st.success("Custom Metal Detector YOLO successful!")
                                image_data = base64.b64decode(response_data['image'])
                                processed_image = Image.open(io.BytesIO(image_data))
                                result = {
                                    'name': uploaded_image.name,
                                    'status': 'success',
                                    'message': "Custom Metal Detector YOLO successful!",
                                    'image': processed_image,
                                    'detections': response_data.get('detections', []),
                                    'processing_time': time.time() - start_time
                                }
                        else:
                            with col1:
                                st.error(f"Request failed with status code {response.status_code}")
                            processed_image = None
                            result = {
                                'name': uploaded_image.name,
                                'status': 'error',
                                'message': f"Request failed with status code {response.status_code}",
                                'image': None,
                                'detections': [],
                                'processing_time': time.time() - start_time
                            }
                    except Exception as e:
                        with col1:
                            st.error(f"Custom YOLOv8 processing failed: {str(e)}")
                        processed_image = None
                        result = {
                            'name': uploaded_image.name,
                            'status': 'error',
                            'message': f"Custom YOLOv8 processing failed: {str(e)}",
                            'image': None,
                            'detections': [],
                            'processing_time': time.time() - start_time
                        }
                
                if result:
                    st.session_state.results.append(result)

    if uploaded_image is not None:
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 2]) 
            with col1:
                st.image(uploaded_image, caption="Uploaded Image", width=800)
            
            if analyze_button and (('processed_image' in locals() and processed_image is not None) or (selected_model == "Model 1: YOLOv8n" and 'processed_image' in locals())):
                with col3:
                    st.image(processed_image, caption="Processed Image", width=800)

    st.markdown("---")
    st.markdown("Made by Sai Ethihas Chanda | currently hosted at http://127.0.0.1:5000")

# Tab 3: Multiple Image Detection
elif tab == "Multiple Image Detection":
    st.markdown('<h1 class="centered-title">Multiple image defect Detector</h1>', unsafe_allow_html=True)
    st.markdown("Upload multiple images to detect defects on metal surfaces using machine learning.")

    st.subheader("Upload Images")
    uploaded_images = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Dropdown box for model selection
    model_options = ["Model 1: YOLOv8n", "Model 2: DSL-YOLO", "Model 3: Custom YOLOv8"]
    selected_model = st.selectbox("Select Detection Model", model_options, on_change=handle_model_selection, args=(st.session_state.get('selected_model', model_options[0]),))

    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            analyze_button = st.button("Analyze Images")
    

    if uploaded_images and analyze_button:
        st.session_state.results = []
        with st.spinner("Analyzing multiple images..."):
            for idx, uploaded_image in enumerate(uploaded_images):
                st.subheader(f"Image {idx + 1}: {uploaded_image.name}")
                with st.container():
                    col1, col2, col3 = st.columns([1, 2, 1])
                    start_time = time.time()
                    processed_image = None
                    result = None
                    if selected_model == "Model 1: YOLOv8n":
                        try:
                            model = get_yolo_model("best_neu.pt")
                            import cv2
                            import numpy as np
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                                img = Image.open(uploaded_image)
                                img.save(tmp_file.name)
                                tmp_file_path = tmp_file.name
                            try:
                                results = model(tmp_file_path)
                                # Convert YOLO output to bytes without creating temp file
                                output_img = results[0].plot()
                                output_pil = Image.fromarray(output_img[..., ::-1])
                                # Save to bytes buffer instead of file
                                buffer = io.BytesIO()
                                output_pil.save(buffer, format="JPEG")
                                buffer.seek(0)
                                processed_image = Image.open(io.BytesIO(buffer.getvalue()))
                                with col1:
                                    st.success("YOLOv8n processing successful!")
                                
                            finally:
                                try:
                                    safe_unlink(tmp_file_path)
                                except Exception as e:
                                    pass  # Silently ignore
                        except Exception as e:
                            with col1:
                                st.error(f"Local YOLOv8n processing failed: {str(e)}")
                        
                        
                    elif selected_model == "Model 2: DSL-YOLO":
                        try:
                            # API call for DSL-YOLO
                            files = {'image': (uploaded_image.name, uploaded_image, uploaded_image.type)}
                            response = requests.post(normal_url, files=files,timeout=300)
                            
                            if response.status_code == 200:
                                with col1:
                                    st.success(f"DSL-YOLO processing successful for {uploaded_image.name}!")
                                processed_image = Image.open(io.BytesIO(response.content))
                                result = {
                                    'name': uploaded_image.name,
                                    'status': 'success',
                                    'message': f"DSL-YOLO processing successful for {uploaded_image.name}!",
                                    'image': processed_image,
                                    'detections': [],
                                    'processing_time': time.time() - start_time
                                }
                            else:
                                with col1:
                                    st.error(f"Request failed for {uploaded_image.name} with status code {response.status_code}")
                                processed_image = None
                                result = {
                                    'name': uploaded_image.name,
                                    'status': 'error',
                                    'message': f"Request failed for {uploaded_image.name} with status code {response.status_code}",
                                    'image': None,
                                    'detections': [],
                                    'processing_time': time.time() - start_time
                                }
                        except Exception as e:
                            with col1:
                                st.error(f"DSL-YOLO processing failed for {uploaded_image.name}: {str(e)}")
                            processed_image = None
                            result = {
                                'name': uploaded_image.name,
                                'status': 'error',
                                'message': f"DSL-YOLO processing failed for {uploaded_image.name}: {str(e)}",
                                'image': None,
                                'detections': [],
                                'processing_time': time.time() - start_time
                            }
                    elif selected_model == "Model 3: Custom YOLOv8":
                        try:
                            # API call for Custom YOLOv8
                            files = {'image': (uploaded_image.name, uploaded_image, uploaded_image.type)}
                            response = requests.post(url, files=files)
                            #response_data = response.json()
                            if response.status_code == 200:
                                
                                with col1:
                                    st.success(f"Custom Metal Detector YOLO successful for {uploaded_image.name}!")
                                processed_image = Image.open(io.BytesIO(response.content))
                                
                                result = {
                                    'name': uploaded_image.name,
                                    'status': 'success',
                                    'message': f"Custom Metal Detector YOLO successful for {uploaded_image.name}!",
                                    'image': processed_image,
                                    'detections':[],
                                    'processing_time': time.time() - start_time
                                }
                            

                        except Exception as e:
                            with col1:
                                st.error(f"Custom YOLOv8 processing failed for {uploaded_image.name}: {str(e)}")
                            processed_image = None
                            result = {
                                'name': uploaded_image.name,
                                'status': 'error',
                                'message': f"Custom YOLOv8 processing failed for {uploaded_image.name}: {str(e)}",
                                'image': None,
                                'detections': None,
                                'processing_time': time.time() - start_time
                            }

                    if result:
                        st.session_state.results.append(result)

                    # Display images
                    with st.container():
                        col1, col2, col3 = st.columns([2, 1, 2])
                        with col1:
                            st.image(uploaded_image, caption=f"Uploaded Image: {uploaded_image.name}", width=800)
                        if processed_image is not None:
                            with col3:
                                st.image(processed_image, caption=f"Processed Image: {uploaded_image.name}", width=800)

# Tab 4: Results
elif tab == "Results":
    st.markdown('<h1 class="centered-title">Results</h1>', unsafe_allow_html=True)
    st.markdown("View detailed results and performance metrics for images processed in the Single and Multiple Image Detection tabs.")

    if not st.session_state.results:
        st.warning("No results available. Please process images in the Single or Multiple Image Detection tabs first.")
    else:
        # Hardcoded 2x3 table
        st.subheader("Overview")
        st.markdown(
            """
            <table class="hardcoded-table">
                <tr>
                    <th>Category</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>YOLOv8n</td>
                    <td>75.2% mAP@50</td>
                </tr>
                <tr>
                    <td>DSL-YOLO</td>
                    <td>79.21.% mAP@50</td>
                </tr>
                <tr>
                    <td>My Approach</td>
                    <td>95.92% mAP@50</td>
                </tr>
            </table>
            """,
            unsafe_allow_html=True
        )
        # Create table data
        table_data = []
        for result in st.session_state.results:
            #defect_classes = ", ".join([d['label'] for d in result['detections']]) if result['detections'] else "N/A"
            table_data.append({
                "Image Name": result['name'],
                "Status": result['status'].capitalize(),
                "Processing Time (s)": f"{result['processing_time']:.2f}"
                
            })

        # Display table
        st.subheader("Detailed Results")
        df = pd.DataFrame(table_data)
        st.dataframe(df)

        # Calculate metrics
        total_images = len(st.session_state.results)
        successful_images = sum(1 for r in st.session_state.results if r['status'] == 'success')
        success_rate = (successful_images / total_images * 100) if total_images > 0 else 0
        avg_processing_time = sum(r['processing_time'] for r in st.session_state.results) / total_images if total_images > 0 else 0

        # Collect defect class distribution (only for Custom YOLOv8)
        defect_counts = {
            'crazing': 0,
            'inclusion': 0,
            'patches': 0,
            'pitted_surface': 0,
            'rolled-in_scale': 0,
            'scratches': 0
        }
        #for result in st.session_state.results:
            #for detection in result['name']:
                #label = detection.get('label')
                #if label in defect_counts:
                    #defect_counts[label] += 1

        # Display metrics
        st.subheader("Summary")
        st.write(f"**Total Images Processed**: {total_images}")
        st.write(f"**Successful Images**: {successful_images} ({success_rate:.2f}% success rate)")
        st.write(f"**Average Processing Time per Image**: {avg_processing_time:.2f} seconds")
        st.subheader("Processing Time per Image")
        image_names = [result['name'] for result in st.session_state.results]
        processing_times = [result['processing_time'] for result in st.session_state.results]

        # Plot processing time data with Plotly
        #st.subheader("Processing Time per Image")
        plot_data = pd.DataFrame({
            'Image Name': [result['name'] for result in st.session_state.results],
            'Processing Time (s)': [result['processing_time'] for result in st.session_state.results]
        })
        fig = px.line(
            plot_data,
            x='Image Name',
            y='Processing Time (s)',
            title='Processing Time per Image',
            markers=True,  # Show data points
            line_shape='linear',  # Straight line segments
            labels={'Image Name': 'Image Name', 'Processing Time (s)': 'Processing Time (s)'}
        )
        fig.update_layout(
            xaxis_title="Image Name",
            yaxis_title="Processing Time (s)",
            xaxis_tickangle=45,  # Rotate x-axis labels for readability
            showlegend=True,
            yaxis=dict(range=[0, None])  # Start y-axis at 0
        )
        fig.update_traces(
            line_color='#36A2EB',  # Blue line color
            marker=dict(size=8)  # Size of data point markers
        )
        st.plotly_chart(fig)

        

    st.markdown("---")
    st.markdown("Made by Sai Ethihas Chanda | currently hosted at http://127.0.0.1:5000")