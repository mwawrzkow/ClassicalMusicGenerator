import os
from time import sleep

try: 
    import eventlet
    eventlet.monkey_patch()
    from flask import Flask, render_template, request, redirect, url_for, send_from_directory
    from flask_socketio import SocketIO, emit
except ImportError:
    os.system("pip install flask Flask-SocketIO eventlet gevent")
    import eventlet
    eventlet.monkey_patch()
    from flask import Flask, render_template, request, redirect, url_for, send_from_directory

import subprocess
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
command = ""
output_path = None  # To store the output directory path

@app.route('/', methods=['GET', 'POST'])
def index():
    global command, output_path  # Use global to track command and output path
    if request.method == 'POST':
        # Extract parameters from the form and create a command to run the model
        gan_type = request.form.get('gan_type')
        epochs = request.form.get('epochs')
        batch_size = request.form.get('batch_size')
        dataset = request.form.get('dataset')
        seq_length = request.form.get('seq_length')
        output_path = request.form.get('output')  # Set the global output_path
        num_generations = request.form.get('num_generations')
        temperature = request.form.get('temperature')
        num_files = request.form.get('num_files')
        
        # Construct the command based on the user input
        command = f'python __main__.py --gan {gan_type} --epochs {epochs} --batch_size {batch_size} '
        command += f'--dataset {dataset} --seq_length {seq_length} --output {output_path} '
        command += f'--length {seq_length} '
        command += f'--temperature {temperature} --num_generations {num_generations} '
        if num_files != '':
            command += f'--num_files {num_files}'
        
        # Redirect to the output page after form submission
        return redirect(url_for('output'))

    return render_template('index.html')

# Route to display the output page and list MIDI files
@app.route('/output')
def output():
    global output_path
    midi_files = []

    # Check if the output_path directory exists and contains MIDI files
    if os.path.isdir(output_path):
        midi_files = [f for f in os.listdir(output_path) if f.endswith('.mid')]

    return render_template('output.html', midi_files=midi_files)

@app.route('/output/<filename>')
def download_file(filename):
    return send_from_directory(output_path, filename)

@app.route('/download-all')
def download_all():
    global output_path
    # check if zip file exists
    if not os.path.exists(os.path.join(output_path, 'output.zip')):
        # create a zip file
        os.system(f'zip -r {output_path}/output.zip {output_path}')
    return send_from_directory(output_path, 'output.zip')

# Monitor output directory for new MIDI files
def monitor_output_directory():
    global output_path
    last_midi_files = set()
    if output_path:
        os.system(f'rm -rf {output_path}/*')

    while True:
        if os.path.isdir(output_path):
            current_midi_files = set([f for f in os.listdir(output_path) if f.endswith('.mid')])

            new_files = current_midi_files - last_midi_files
            if new_files:
                # Emit new MIDI files to the client
                socketio.emit('new_midi_files', list(new_files))
                last_midi_files = current_midi_files

        # sleep using eventlet to yield control to the event loop
        eventlet.sleep(1)

# Run the command in a separate green thread when the WebSocket connects
@socketio.on('connect')
def handle_connect():
    print("Client connected")

    def run_command():
        global command, output_path  # Use global command that was set in index()
        if command:
            socketio.emit('output', {'data': 'Running the command...'})
            socketio.emit('output', {'data': command})

            # delete all files in the output directory
            last_line = ""
            # if last line contains Successfully installed then restart process
            while True:
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                for line in iter(process.stdout.readline, b''):
                    socketio.emit('output', {'data': line.decode('utf-8')})  # Emit output to client
                    last_line = line.decode('utf-8')
                    eventlet.sleep(0)  # Yield control to the event loop for non-blocking operation
                process.stdout.close()
                process.wait()
                if 'Successfully installed' not in last_line:
                    break
            socketio.emit('output', {'data': 'Finished!'})

    eventlet.spawn_n(run_command)
    eventlet.spawn_n(monitor_output_directory)
@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

if __name__ == '__main__':
    # Start the directory monitoring thread
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, use_reloader=False)
