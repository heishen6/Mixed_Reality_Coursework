from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import os
import datetime
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='static')
CORS(app)  # 允许跨域请求，适用于开发环境

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')
# 获取当前脚本文件的目录路径
base_dir = os.path.dirname(os.path.abspath(__file__))
path_to_power_image = os.path.join(base_dir, 'power.png')
path_to_crystal_image = os.path.join(base_dir, 'te.png')

# 检查并加载模板图片
if not os.path.exists(path_to_power_image) or not os.path.exists(path_to_crystal_image):
    print("One or both template images not found. Please check the file paths.")
    exit(1)

template_power = cv2.imread(path_to_power_image, cv2.IMREAD_GRAYSCALE)
template_crystal = cv2.imread(path_to_crystal_image, cv2.IMREAD_GRAYSCALE)

if template_power is None or template_crystal is None:
    print("Failed to load one or more template images.")
    exit(1)

orb = cv2.ORB_create(10000)
kp_power, des_power = orb.detectAndCompute(template_power, None)
kp_crystal, des_crystal = orb.detectAndCompute(template_crystal, None)

def match_image(uploaded_image):
    uploaded_gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)
    kp_uploaded, des_uploaded = orb.detectAndCompute(uploaded_gray, None)
    if des_uploaded is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_power = bf.match(des_uploaded, des_power) if des_power is not None else []
    matches_crystal = bf.match(des_uploaded, des_crystal) if des_crystal is not None else []

    good_matches_power = [m for m in matches_power if m.distance < 50]
    good_matches_crystal = [m for m in matches_crystal if m.distance < 50]

    print(f"Number of good matches for power image: {len(good_matches_power)}")
    print(f"Number of good matches for crystal image: {len(good_matches_crystal)}")

    if len(good_matches_power) > len(good_matches_crystal):
        return 98
    elif len(good_matches_crystal) > len(good_matches_power):
        return 26
    else:
        return None

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    temp_path = os.path.join('/tmp', filename)
    file.save(temp_path)
    uploaded_image = cv2.imread(temp_path)
    os.remove(temp_path)
    match_result = match_image(uploaded_image)
    if match_result is not None:
        return jsonify({'decodedNumber': match_result}), 200
    else:
        return jsonify({'error': 'No matching image found'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5080, debug=True)
