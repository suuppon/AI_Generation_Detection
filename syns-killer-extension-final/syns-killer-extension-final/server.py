# conda activate chrome
# server.py
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os, json
from natsort import natsorted  # pip install natsort

app = Flask(__name__)
CORS(app)

# ✅ 여기를 원하는 경로로 바꾸세요
IMAGE_DIR = r"/Users/kimsoojin/Desktop/SKKU/AIchampion/syns-killer-extension-copy/pilot/IMG"   # 예: "D:/work/images"
JSON_FILE = r"/Users/kimsoojin/Desktop/SKKU/AIchampion/syns-killer-extension-copy/pilot/DATA/25-09-02.json"

@app.get("/overlay/list")
def overlay_list():
    files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif"))
    ]
    files = natsorted(files)  # 1.jpg, 2.jpg, 10.jpg 순으로 정렬
    return jsonify({"files": files})

@app.get("/overlay/files/<path:filename>")
def overlay_file(filename):
    return send_from_directory(IMAGE_DIR, filename)

@app.get("/read-json")
def read_json():
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)


