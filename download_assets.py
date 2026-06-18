import os
import requests
import sys

# Define model paths and URLs
MODEL_DIR = os.path.join(os.path.dirname(__file__), "assets", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion-ferplus-8.onnx")

# List of URLs to try in order (fallback mechanism)
MODEL_URLS = [
    "https://github.com/spmallick/learnopencv/raw/master/Facial-Emotion-Recognition/emotion-ferplus-8.onnx",
    "https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
]

def download_file(urls, dest_path, progress_callback=None):
    """Downloads a file from a list of URLs with fallback support and progress bar."""
    if os.path.exists(dest_path):
        print(f"Model already exists at: {dest_path}")
        return True

    # Ensure output directory exists
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    success = False
    for url in urls:
        print(f"Attempting to download from: {url}")
        try:
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024 * 1024  # 1MB
                downloaded = 0
                
                with open(dest_path, 'wb') as file:
                    for data in response.iter_content(block_size):
                        file.write(data)
                        downloaded += len(data)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            sys.stdout.write(f"\rProgress: [{downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB] ({percent:.1f}%)")
                            sys.stdout.flush()
                            if progress_callback:
                                progress_callback(downloaded, total_size)
                        else:
                            sys.stdout.write(f"\rProgress: [{downloaded/(1024*1024):.1f}MB downloaded]")
                            sys.stdout.flush()
                            if progress_callback:
                                progress_callback(downloaded, 0)
                print("\nDownload completed successfully!")
                success = True
                break
            else:
                print(f"Failed with status code: {response.status_code}")
        except Exception as e:
            print(f"Error occurred during download: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path) # Clean up partial download
    
    return success

def setup_assets():
    """Sets up the required directories and downloads the model."""
    print("Setting up system assets...")
    success = download_file(MODEL_URLS, MODEL_PATH)
    if not success:
        print("\nERROR: Failed to download the emotion classification model after trying all mirrors.")
        print("Please check your internet connection and try running the script again.")
        sys.exit(1)
    print("Assets setup completed successfully!")

if __name__ == "__main__":
    setup_assets()
