import requests
import pandas as pd

# Path to the image you want to upload
image_path = 'svmtest/pitted_surface_170.jpg'  # Change this to your test image

# Endpoint URL
url = 'http://127.0.0.1:5000/upload'

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
