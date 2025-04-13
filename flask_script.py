from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import numpy as np 
from flask import send_file
from io import BytesIO
from PIL import Image
import glob
import cv2
import os

import pandas as pd
from skimage.filters import sobel
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from tqdm import tqdm
from ultralytics import YOLO
import torch
import joblib
# === Load YOLO model and setup hook ===
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = YOLO("best_neu.pt")
features = []
def hook(module, input, output):
    features.append(output)
hook_handle = model.model.model[-2].register_forward_hook(hook)  # Hook to last feature layer

# === Class names (update to your own)
CLASS_NAMES = ["crazing", "inclusion","patches","pitted_surface","rolled-in_scale","scratches"]

# === IoU calculator ===
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# === Parse YOLO TXT annotation ===
def parse_yolo_txt_annotation(txt_path, img_width, img_height):
    gt_boxes = []
    gt_labels = []
    if not os.path.exists(txt_path):
        return gt_boxes, gt_labels
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            w = float(parts[3]) * img_width
            h = float(parts[4]) * img_height
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)
            gt_boxes.append([x1, y1, x2, y2])
            gt_labels.append(CLASS_NAMES[class_id])
    return gt_boxes, gt_labels

# === GLCM feature extractor ===
def extract_glcm_features(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = np.array(img)

    image_dataset = pd.DataFrame()

    

    df = pd.DataFrame()
    GLCM = graycomatrix(img, [1], [0])       
    GLCM_Energy = graycoprops(GLCM, 'energy')[0]
    df['Energy'] = GLCM_Energy
    GLCM_corr = graycoprops(GLCM, 'correlation')[0]
    df['Corr'] = GLCM_corr       
    GLCM_diss = graycoprops(GLCM, 'dissimilarity')[0]
    df['Diss_sim'] = GLCM_diss       
    GLCM_hom = graycoprops(GLCM, 'homogeneity')[0]
    df['Homogen'] = GLCM_hom       
    GLCM_contr = graycoprops(GLCM, 'contrast')[0]
    df['Contrast'] = GLCM_contr


    GLCM2 = graycomatrix(img, [3], [0])       
    GLCM_Energy2 = graycoprops(GLCM2, 'energy')[0]
    df['Energy2'] = GLCM_Energy2
    GLCM_corr2 = graycoprops(GLCM2, 'correlation')[0]
    df['Corr2'] = GLCM_corr2       
    GLCM_diss2 = graycoprops(GLCM2, 'dissimilarity')[0]
    df['Diss_sim2'] = GLCM_diss2       
    GLCM_hom2 = graycoprops(GLCM2, 'homogeneity')[0]
    df['Homogen2'] = GLCM_hom2       
    GLCM_contr2 = graycoprops(GLCM2, 'contrast')[0]
    df['Contrast2'] = GLCM_contr2

    GLCM3 = graycomatrix(img, [5], [0])       
    GLCM_Energy3 = graycoprops(GLCM3, 'energy')[0]
    df['Energy3'] = GLCM_Energy3
    GLCM_corr3 = graycoprops(GLCM3, 'correlation')[0]
    df['Corr3'] = GLCM_corr3       
    GLCM_diss3 = graycoprops(GLCM3, 'dissimilarity')[0]
    df['Diss_sim3'] = GLCM_diss3       
    GLCM_hom3 = graycoprops(GLCM3, 'homogeneity')[0]
    df['Homogen3'] = GLCM_hom3       
    GLCM_contr3 = graycoprops(GLCM3, 'contrast')[0]
    df['Contrast3'] = GLCM_contr3

    GLCM4 = graycomatrix(img, [0], [np.pi/4])       
    GLCM_Energy4 = graycoprops(GLCM4, 'energy')[0]
    df['Energy4'] = GLCM_Energy4
    GLCM_corr4 = graycoprops(GLCM4, 'correlation')[0]
    df['Corr4'] = GLCM_corr4       
    GLCM_diss4 = graycoprops(GLCM4, 'dissimilarity')[0]
    df['Diss_sim4'] = GLCM_diss4       
    GLCM_hom4 = graycoprops(GLCM4, 'homogeneity')[0]
    df['Homogen4'] = GLCM_hom4       
    GLCM_contr4 = graycoprops(GLCM4, 'contrast')[0]
    df['Contrast4'] = GLCM_contr4
    
    GLCM5 = graycomatrix(img, [0], [np.pi/2])       
    GLCM_Energy5 = graycoprops(GLCM5, 'energy')[0]
    df['Energy5'] = GLCM_Energy5
    GLCM_corr5 = graycoprops(GLCM5, 'correlation')[0]
    df['Corr5'] = GLCM_corr5       
    GLCM_diss5 = graycoprops(GLCM5, 'dissimilarity')[0]
    df['Diss_sim5'] = GLCM_diss5       
    GLCM_hom5 = graycoprops(GLCM5, 'homogeneity')[0]
    df['Homogen5'] = GLCM_hom5       
    GLCM_contr5 = graycoprops(GLCM5, 'contrast')[0]
    df['Contrast5'] = GLCM_contr5

    df =df.drop(["Corr4","Diss_sim4","Contrast4","Corr5","Diss_sim5","Homogen3","Homogen4","Homogen5","Contrast5","Energy5"],axis=1)
    return df.values.flatten()


# === YOLO feature extractor ===

def extract_yolo_features(image):
    im = cv2.resize(image, (640, 640))
    im = im.astype(np.float32) / 255.0
    im = torch.from_numpy(im.transpose(2, 0, 1)).unsqueeze(0)

    # ✅ Match device with model (GPU or CPU)
    device = next(model.model.parameters()).device
    im = im.to(device)

    features.clear()
    with torch.no_grad():
        _ = model.model(im)

    feat_tensor = features[0].cpu()  # optionally move to CPU before converting to numpy
    feat_vector = torch.nn.functional.adaptive_avg_pool2d(feat_tensor, 1).view(feat_tensor.shape[0], -1)
    return feat_vector.squeeze().numpy()

def draw_boxes_with_labels(img_path, det_boxes, labels, box_color=(0, 255, 0), text_color=(255, 255, 255)):
    print("labels are hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    #print(list(labels[0]))
    img = cv2.imread(img_path)
    img_copy = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2

    for i, (x1, y1, x2, y2) in enumerate(det_boxes):
        label = str(labels[i])#labels[i]

        # Draw rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), box_color, thickness)

        # Put label text
        cv2.putText(img_copy, label, (x1, y1 - 10), font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    #img_copy.save("test.jpg")

    return img_copy

# === Thermal conversion ===
def convert_to_thermal(img_gray):
    return cv2.applyColorMap(img_gray, cv2.COLORMAP_INFERNO)

# === Main pipeline ===
@app.route('/upload', methods=['POST'])
def process_images():
    bounding_box_dict = {}
    image_folder = "uploads"
    os.makedirs(image_folder, exist_ok=True)
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    image_folder = os.path.join(UPLOAD_FOLDER,filename)
    print("LOOOOOOOOOOOOOOOOOK HEEEEEEEEEEEEEEEEEEEEEEEEEEEREEEEEEEEE")
    print(image_folder)
    file.save(image_folder)
    all_data = []
    all_targets = []

    for file in tqdm(os.listdir("uploads"), desc="Processing"):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        filename = os.path.splitext(file)[0]
        img_path = os.path.join(image_folder)
        #annot_path = os.path.join(annotation_folder, filename + ".txt")

        original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            continue

        h, w = original.shape
        thermal = convert_to_thermal(original)
        original_color = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

        #gt_boxes, gt_labels = parse_yolo_txt_annotation(annot_path, w, h)

        results = model(original_color, conf=0.5)
        print("________________________________________________________________________")
        print(results)
        det_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

        for x1, y1, x2, y2 in det_boxes:
            bounding_box_dict[str(x1)+str(y1)+str(x2)+str(y2)]=""
            norm_crop = original_color[y1:y2, x1:x2]
            thermal_crop = thermal[y1:y2, x1:x2]
            if norm_crop.size == 0 or thermal_crop.size == 0:
                continue

            try:
                yolo_vec = extract_yolo_features(norm_crop)
                glcm_vec = extract_glcm_features(thermal_crop)
                row = list(yolo_vec) + list(glcm_vec)
                all_data.append(row)

                # Match with closest GT box
                

            except Exception as e:
                print(f"⚠️ Error in {file}: {e}")

    # Build DataFrame
    yolo_cols = [f"{i}" for i in range(len(yolo_vec))]
    # Column names for GLCM features
    glcm_cols = ['Energy', 'Corr', 'Diss_sim', 'Homogen', 'Contrast',
             'Energy2', 'Corr2', 'Diss_sim2', 'Homogen2', 'Contrast2',
             'Energy3', 'Corr3', 'Diss_sim3', 'Contrast3', 'Energy4']
    df = pd.DataFrame(all_data, columns=yolo_cols + glcm_cols)

    

    # 2. Combine feature vectors into a row
    #row = list(yolo_vec) + list(glcm_vec)

# 3. When all rows are ready, construct DataFrame
    df = pd.DataFrame(all_data, columns=yolo_cols + glcm_cols)
    #df = pd.DataFrame(all_data)
    print("✅ Final feature matrix shape:", df.shape)
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    svm_model = joblib.load("svm_model.pkl")
    le = joblib.load("label_encoder.pkl")
    
    X_new = df.drop(columns=["target"], errors='ignore')

    # Scale and reduce dimensionality
    X_scaled = scaler.transform(X_new)
    X_pca = pca.transform(X_scaled)
    #print(X_pca.head())
    # Predict using the trained SVM model
    predicted_classes = svm_model.predict(X_pca)

    # Add to the dataframe
    df["svm_pred"] = predicted_classes
    df["svm_pred_label"] = le.inverse_transform(df["svm_pred"])
    
    for i,(x1, y1, x2, y2) in enumerate(det_boxes):
        
        bounding_box_dict[str(x1)+str(y1)+str(x2)+str(y2)]=df["svm_pred_label"].iloc[i]
    final_img = draw_boxes_with_labels(image_folder,det_boxes, list(bounding_box_dict.values()))
    #print(final_img)

    print("✅ Predictions added to dataframe!")
    image_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # Save image to in-memory buffer
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)

    # Return as file
    return send_file(buffer, mimetype='image/png', as_attachment=False, download_name='eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee.png')

# === Run pipeline ===
#df_combined = process_images("images/Testing_data")
@app.route('/')
def index():
    return "YOLO + GLCM Flask API is running!"

if __name__ == '__main__':
    app.run(debug=True)