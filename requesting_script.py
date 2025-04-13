import requests
import pandas as pd

# Path to the image you want to upload
image_path = 'images/Testing_data/scratches_3.jpg'  # Change this to your test image

# Endpoint URL
url = 'http://127.0.0.1:5000/upload'

# Send POST request with image
with open(image_path, 'rb') as img_file:
    files = {'image': img_file}
    response = requests.post(url, files=files)

# Check if the request was successful
if response.status_code == 200:
    print("Request successful!")

    # Parse the JSON response
    response_data = response.json()
    
    # Check if the response contains the 'features' key
    if 'features' in response_data:
        # Convert the 'features' list of dictionaries into a DataFrame
        df = pd.DataFrame(response_data['features'])
        print("DataFrame:")
        print(df)
    else:
        print("No features found in the response.")
else:
    print(f"Request failed with status code {response.status_code}")
