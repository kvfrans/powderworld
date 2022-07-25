import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

from flask import Flask
from flask import send_file
from flask import render_template
from flask import request
from flask_socketio import SocketIO, emit
import sys
import io
import base64
import time
import imageio
from PIL import Image
from multiprocessing import Process, Queue, Manager
from ctypes import c_wchar_p


from powderworld import PowderWorld

app = Flask(__name__)
socketio = SocketIO(app)



# ==================== GAME LOOP ===============
def update_process(q, sharedMessage):
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        print("Waiting for start signal")
        # _ = q.get()
        print("Start!")

        pw = PowderWorld(device)

        def reset_world():
            new_world = torch.zeros((1, pw.NUM_CHANNEL, 128, 128), dtype=torch.float32, device=device)
            pw.add_element(new_world[:, :, :, :], "wall")
            pw.add_element(new_world[:, :, 1:-1, 1:-1], "empty")
            pw.add_element(new_world[:, :, 10:20, 30:40], "sand")

            wind = 5 * torch.Tensor([1, 0]).to(device)[None,:,None,None]
            pw.add_element(new_world[:, :, 40-2:40+2, 40-2:40+2], 'wind', wind)

            wind = 5 * torch.Tensor([-1, 0]).to(device)[None,:,None,None]
            pw.add_element(new_world[:, :, 80-2:80+2, 40-2:40+2], 'wind', wind)

            return new_world

        world = reset_world()
        world = pw(world)
        pw_jit = pw
        img = pw.render(world)
        print("First step OK")

        last_wind = [0,0]

        while True:
            # STEP
            # emit('message', "Hi", broadcast=True)
            if not q.empty():
                action = q.get()
                if action["action"] == "reset":
                    world = reset_world()
                else:
                    block = action["action"]
                    x = action["x"]
                    y = action["y"]
                    windx = action["windx"]
                    windy = action["windy"]
                    wind = 5 * torch.Tensor([windy, windx]).to(device)[None,:,None,None]
                    pw.add_element(world[:, :, y-3:y+3, x-3:x+3], block, wind)
            else:
                world = pw(world)
                img = pw.render(world)

                im = Image.fromarray(img.astype("uint8"))
                # im = im.resize((128 * 6, 128 * 6), Image.NEAREST)
                buffer = io.BytesIO()
                im.save(buffer,format="PNG")
                myimage = buffer.getvalue()                     
                dat = "data:image/png;base64,"+base64.b64encode(myimage).decode()
                sharedMessage.value = dat

                time.sleep(0.005)
    

    
# ================== SERVER LOGIC =================
if __name__ == '__main__':
    manager = Manager()
    sharedMessage = manager.Value(c_wchar_p, "Blank")
    
    q = Queue()
    p = Process(target=update_process, args=(q,sharedMessage))
    p.start()


    @app.route("/")
    def hello_world():
        q.put({"action": "reset"})
        return render_template('server_page.html')

    @app.route("/current_render")
    def current_render():
        return send_file('server_render.png', mimetype='image/png')

    @app.route('/make_block', methods=['POST'])
    def make_block():
        print(request.form)
        x = float(request.form['x'])
        y = float(request.form['y'])
        windx = float(request.form['windx'])
        windy = float(request.form['windy'])
        action = request.form['action']
        x = min(max(int(x*128), 4), 128-4)
        y = min(max(int(y*128), 4), 128-4)
        q.put({"action": action, "x": x, "y": y, "windx": windx, "windy": windy})
        return ""
    
    @socketio.on('message')
    def handle_message(message):
        emit('blockdata', sharedMessage.value)


    socketio.run(app, host="0.0.0.0")
