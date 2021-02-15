from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, join_room
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

with open('intents.json','r', encoding='utf-8') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)



input_size = data["input_size"]
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words= data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size)

model.load_state_dict(model_state)
model.eval()


bot_name = "Ray's RoboAdviser"

app = Flask(__name__)
socketio = SocketIO(app, logger = True)


@app.route('/')
def home():
    return render_template("chat.html")


@socketio.on('send_message')
def handle_send_message_event(data):

    socketio.emit('recieve_message', data)
    print(data['message'])

    
    global sentence = data['message']
    

    global sentence = tokenize(sentence)

    X = bag_of_words(sentence, all_words)

    X= X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)

    _, predicted = torch.max(output,dim =1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:

        for intent in intents['intents']:
            if tag == intent["tag"]:
                reply = intent['responses'][0]
                #print(f"{bot_name}: {intent['responses'][0]}")
                
                print(reply)


    else:
        reply = "I do not understand"
        #print(f'{bot_name}: I do not understand')
        
    

@socketio.on('reply_message')
def handle_reply_message_event(data):

    socketio.emit('reply_message', data)
    
    




if __name__ == '__main__':
    socketio.run(app, debug=True)
    #SocketIO(app,cors_allowed_origins="*")
