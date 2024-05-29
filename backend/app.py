from flask import Flask, request, jsonify, send_from_directory, render_template
import subprocess
import json
import pyudev
import os

app = Flask(__name__, static_folder='../frontend/dist', static_url_path='/')
started = False
# Load initial configuration from file


CONFIG_FILE = 'config.json'
TEST_VIDEOS_FOLDER = 'test_files'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_FOLDER = os.path.join(BASE_DIR, '../frontend/public/eventos')
with open('config.json', 'r') as f:
    config = json.load(f)

# Simular una base de datos de usuarios
users = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

def list_video_files():
    return [f for f in os.listdir(TEST_VIDEOS_FOLDER) if f.endswith(('.mp4', '.avi', '.mkv'))]

def list_camera_devices():
    context = pyudev.Context()
    devices = []
    for index, device in enumerate(context.list_devices(subsystem='video4linux')):
        device_name = device.get('ID_V4L_PRODUCT')
        device_path = device.device_node
        if device_name and device_path:
            devices.append({'name': f"{device_name} ({device_path})", 'index': index})
    return devices


@app.route('/eventos')
def list_media():
    files = os.listdir(MEDIA_FOLDER)
    media_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg', '.mp4', '.avi', '.mkv'))]
    return jsonify(media_files)

@app.route('/eventos/<filename>')
def get_media(filename):
    print("ACA ESTOY: ", filename)
    try:
        return send_from_directory(MEDIA_FOLDER, filename)
    except Exception as e:
        print(e)
        return str(e), 404

@app.route('/upload', methods=['PUT'])
def upload_image():
    if 'image' not in request.files:
        return "No image part", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    # Guardar la imagen en el servidor
    file.save(f"./uploads/{file.filename}")
    return "Image uploaded successfully", 200

@app.route('/users', methods=['GET'])
def get_users():
    return jsonify(users), 200

@app.route('/settings', methods=['POST'])
def set_settings():
    setting = request.json.get('setting')
    if not setting:
        return "Invalid setting", 400

    # Guardar la configuración en el servidor (simulado)
    print(f"Setting saved: {setting}")
    return "Setting saved successfully", 200

def execute_makefile_rule(rule):
    # Execute the specified rule from the Makefile
    result = subprocess.run(['make', '-f', "Makefile", rule], capture_output=True, text=True)
    print(result.returncode)
    return (result.stdout.strip(), result.returncode) if result.returncode == 0 else (result.stderr.strip(), result.returncode)

@app.route('/toggle_command', methods=['POST'])
def toggle_command():
    global started
    if started:
        output, started = execute_makefile_rule('stop')
        started = bool(started)
    else:
        output, started = execute_makefile_rule('start')
        started = bool(started == 0)
    print("output: ", output)
    return jsonify({'output': output, 'started': started})

@app.route('/get_config', methods=['GET'])
def get_config():
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    video_files = list_video_files()
    camera_devices = list_camera_devices()
    return jsonify({
        'config': config,
        'video_files': video_files,
        'camera_devices': camera_devices
    })


@app.route('/save_config', methods=['POST'])
def save_config():
    new_config = request.json
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    
    # Update paths for video files
    if 'front_video_input' in new_config:
        if new_config['front_video_input'] in list_video_files():
            new_config['front_video_input'] = os.path.join(TEST_VIDEOS_FOLDER, new_config['front_video_input'])
        else:
            new_config['front_video_input'] = int(new_config['front_video_input'])
    if 'side_video_input' in new_config:
        if new_config['side_video_input'] in list_video_files():
            new_config['side_video_input'] = os.path.join(TEST_VIDEOS_FOLDER, new_config['side_video_input'])
        else:
            new_config['side_video_input'] = int(new_config['side_video_input'])
    config.update(new_config)

    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    return jsonify({'status': 'success'})


@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(port=5000)
