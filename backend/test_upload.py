import requests
import io
import os
import uuid

BASE_URL = "http://127.0.0.1:8000"
USER_ID = "test-user-debug-123"

def test_upload():
    print(f"Testing upload for user: {USER_ID}")
    
    pdf_path = "scanned_dummy.pdf"
    if not os.path.exists(pdf_path):
        print(f"Could not find {pdf_path}")
        return

    with open(pdf_path, 'rb') as f:
        files = {'file': (pdf_path, f, 'application/pdf')}
        headers = {'X-User-ID': USER_ID}
        
        try:
            res = requests.post(f"{BASE_URL}/upload", headers=headers, files=files)
            print(f"Status Code: {res.status_code}")
            print(f"Response: {res.text}")
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    test_upload()
