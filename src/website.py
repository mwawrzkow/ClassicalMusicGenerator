import os
from time import sleep

try: 
    import eventlet
    eventlet.monkey_patch()
    from flask import Flask, render_template, request, redirect, url_for, send_from_directory
    from flask_socketio import SocketIO, emit
    from flask_cors import CORS
except ImportError:
    os.system("pip install flask Flask-SocketIO eventlet gevent flask-cors")
    import eventlet
    eventlet.monkey_patch()
    from flask import Flask, render_template, request, redirect, url_for, send_from_directory
    from flask_socketio import SocketIO, emit
    from flask_cors import CORS

import subprocess
import threading
from datetime import datetime

app = Flask(__name__, static_folder="dist")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global variables
command = ""
output_path = None  # To store the output directory path
process = None  # To store the process object
logs = [{
    'message': "Welcome to the Music Generator",
    'event': "INFO",
    'timestamp': "0000-00-00 00:00:00"
    }]  # To store the logs

@app.route('/_nuxt/<path:filename>')
def nuxt_static(filename):
    return send_from_directory(os.path.join(app.static_folder, '_nuxt'), filename)

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route("/generate", methods=['POST'])
def generate():
    global command
    global process
    global logs
    global output_path
    logs = []
    print("Generating music...")
    print(request.json)
    # prepare the command
    #json {'model_name': 'WGan', 'num_generations': 10, 'midi_dir': '', 'temperature': '1.0', 'sequence_length': 50}
    # example command python __main__.py             "args": ["--gan", "rnn", "--epochs", "150", "--batch_size", "512", "--dataset", "midis_v1.2/midis", "--seq_length", "50", "--num_files", "1000", "--output", "otp_rnn_final_v3_last_try", "--length", "50", "--temperature", "0.5", "--num_generations", "100","--checkpoint", "model_data/rnn_last_try.keras" ],
    model = request.json['model_name'] if request.json['model_name'] != "WGan" else "gan"
    model = model.lower()
    num_generations = request.json['num_generations']
    midi_dir = request.json['midi_dir']
    temperature = request.json['temperature']
    sequence_length = request.json['sequence_length']
    midi_dir = midi_dir if midi_dir else "midi_data/MIDIs"
    num_files = request.json['num_files'] if request.json['num_files'] != 0 else len(os.listdir(midi_dir))
    command = f"python __main__.py --gan {model} --epochs 0 --num_generations {num_generations} --dataset {midi_dir} --temperature {temperature} --seq_length {sequence_length} --length {sequence_length} --output output"
    model_checkpoint = "model_data/rnn_last_try.keras" if model == "rnn" else "model_data/"
    command = f"{command} --checkpoint {model_checkpoint} --dataset {midi_dir} --num_files {num_files}"
    if not process:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Ensures the output is in string format
            bufsize=1,  # Line-buffered
            universal_newlines=True
        )
    output_path = "output"
    # Start a background task to read the subprocess output
    socketio.start_background_task(target=read_process_output, p=process)
    return "Generating music..."

@app.route("/download", methods=['GET'])
def download():
    global output_path
    # zip the output directory
    zip_file = f"{output_path}.zip"
    os.system(f"zip -r {zip_file} ./{output_path}")
    return send_from_directory(os.getcwd(), zip_file, as_attachment=True)

@app.route("/logs", methods=['GET'])
def get_logs():
    # return logs as logs.log file
    with open("logs.log", "w") as f:
        for log in logs:
            f.write(f"{log['timestamp']} {log['event']} {log['message']}\n")
    return send_from_directory(os.getcwd(), "logs.log", as_attachment=True)
def read_process_output(p):
    """Read the subprocess stdout and stderr and emit to clients."""
    global command
    global logs
    emit_log('INFO', f'Running command: {command}')
    def read_stream(stream, event_type):
        for line in iter(stream.readline, ''):
            if line:
                emit_log(event_type, line.strip())
            sleep(0.01)
        stream.close()
    
    # Start green threads to read stdout and stderr
    socketio.start_background_task(target=read_stream, stream=p.stdout, event_type='INFO')
    socketio.start_background_task(target=read_stream, stream=p.stderr, event_type='ERROR')
    
    p.wait()
    # get process return code
    return_code = p.returncode
    emit_log('INFO', f"Process finished. Return code: {return_code}")
    p.stdout.close()
    p.stderr.close()
    global process
    process = None

def emit_log(event_type, message):
    """Emit a log message to all connected clients in the '/ws' namespace."""
    payload =  {
            'timestamp': get_current_timestamp(),
            'event': event_type,
            'message': message
        }
    socketio.emit(
        'status',
       payload,
        namespace='/ws'
    )
    global logs
    return payload
def get_current_timestamp():
    """Return the current timestamp as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@app.route("/train", methods=['POST'])
def train():
    global logs
    global command
    global process
    global output_path
    logs = []
    
    print("Training model...")
    print(request.json)
    
    # Extracting parameters from the request JSON
    model = request.json['model_name'].lower() 
    num_generations = request.json['num_generations']
    midi_dir = request.json['midi_files_dir']
    temperature = request.json['temperature']
    sequence_length = request.json['num_sequences']
    midi_dir = midi_dir if midi_dir else "midi_data/MIDIs"
    num_files = request.json['num_files'] if request.json['num_files'] != 0 else len(os.listdir(midi_dir))
    num_epochs = request.json['num_epochs']
    
    # Construct the command without requiring a checkpoint
    command = f"python __main__.py --gan {model} --epochs {num_epochs} --num_generations {num_generations} --dataset {midi_dir} --temperature {temperature} --seq_length {sequence_length} --length {sequence_length} --output output"
    command = f"{command} --dataset {midi_dir} --num_files {num_files}"
    output_path = "output"
    emit_log('INFO', f"Running command: {command}")
    print(f"Running command: {command}")
    
    if not process:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Ensures the output is in string format
            bufsize=1,  # Line-buffered
            universal_newlines=True
        )
    
    # Start a background task to read the subprocess output
    socketio.start_background_task(target=read_process_output, p=process)
    
    return "Training model..."

# check if client gets tonnected to /ws endpoint and if so, open websocket connection
@socketio.on('connect', namespace='/ws')
def ws_connect():
    print('Client connected')
    print("We've been distributed to a worker")    
    emit('status', {'data': 'Connected'})
    for log in logs:
        emit_log('INFO', log)

if __name__ == '__main__':
    # Start the directory monitoring thread
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, use_reloader=True)