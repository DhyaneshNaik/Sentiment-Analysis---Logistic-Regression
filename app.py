
import pandas as pd
import pickle
import numpy as np
from flask import *

from methods import *

app = Flask(__name__)

frquencies = load_frequencies('freq')
thetas = np.load('./Freq/theta.npy')


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predict", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        text = request.form['userInput']
        value = predict_tweet(text, frquencies, thetas)
        image_name = ''
        if value > 0.5:
            image_name = 'Positive.jpg'
            color='#90EE90'
        else:
            image_name = 'Negative.jpg'
            color='#ff3f34'

        return render_template('predict.html',
                               text=text,
                               image=image_name,color=color)
    return None
    
if __name__ == "__main__":
    app.run(debug=True)