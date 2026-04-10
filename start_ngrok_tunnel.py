#!/usr/bin/env python3
"""
Start ngrok tunnel for Flask server
"""
from pyngrok import ngrok
import time

# Connect to Flask server running on port 5000
public_url = ngrok.connect(5000)
print(f"\n{'='*60}")
print(f"🎉 Flask is now publicly accessible!")
print(f"{'='*60}")
print(f"Public URL: {public_url}")
print(f"{'='*60}\n")
print("Use this URL in your Streamlit app's FLASK_URL environment variable")
print("Or update requesting_script.py with this URL\n")

# Keep tunnel alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down ngrok tunnel...")
    ngrok.kill()
