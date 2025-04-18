import requests
import pandas as pd
import os
import io
import zipfile

# Path to the image you want to upload
image_path = 'Data/NEU-DET/NEU-DET/IMAGES/crazing_1.jpg'  # Change this to your test image
folder_path = 'svmtest'

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

# Endpoint URL
url = 'http://127.0.0.1:5000/Metal_surface_pred'
test_url = 'http://127.0.0.1:5000/'
folder_url = 'http://127.0.0.1:5000/Metal_surface_pred_folder'

test_response=requests.get(test_url)
print(test_response.content)

# Send POST request with image
with open(image_path, 'rb') as img_file:
    files = {'image': img_file}
    response = requests.post(url, files=files)


# Check if the request was successful
if response.status_code == 200:
    print("Request successful!")
    with open('kkkkkkkkkkkkkkkkkkkk.png', 'wb') as f:
        f.write(response.content)

    # Parse the JSON response
    
    
    # Check if the response contains the 'features' key
    
else:
    print(f"Request failed with status code {response.status_code}")

#zip_data = zip_folder_in_memory(folder_path)
#files = {'folder': ('svmtest.zip', zip_data, 'application/zip')}
#folder_response = requests.post(folder_url, files=files)
#print(folder_response.content)

#if response.status_code == 200:
   # extract_folder = 'processed_images'
 #   os.makedirs(extract_folder, exist_ok=True)

  #  with zipfile.ZipFile(io.BytesIO(response.content)) as z:
  #      z.extractall(extract_folder)

   # print(f"✅ Extracted and saved to '{extract_folder}'")
#else:
    #print(f"❌ Request failed with status code {response.status_code}")
