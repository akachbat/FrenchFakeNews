import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import urllib
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
from newspaper import Article

#model params
MAX_LEN = 500

app = Flask(__name__)

#Load the tokenizer
with open('models/tokenizer.pickle', 'rb') as text_tokenizer:  
    text_tokenizer = pickle.load(text_tokenizer)
    
model = keras.models.load_model('models')    

@app.route('/')
def home():
    print('Home page start!')
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def predict():

    url = request.args.get('url','#')
    url = urllib.parse.unquote(url) 
    print('URL: ', url)
    article = get_article(url)
    prediction, score = get_prediction(article.text)
    print('prediction {0}, score {1}'.format(prediction, score))
    score = round(score * 100, 2)
    
    return render_template('check.html', article=article, prediction=prediction, score=score)

@app.route('/results',methods=['POST'])
def results():

    #data = request.get_json(force=True)
    #prediction = model.predict([np.array(list(data.values()))])

    output =10
    return jsonify(output)


def get_prediction(text):
    # Preprocessing
    text_np_array = text_tokenizer.texts_to_sequences([text])
    text_np_array = sequence.pad_sequences(text_np_array, maxlen=MAX_LEN, padding="post", value=0)
    # Prediction
    score = model.predict(text_np_array)[0][0]
    #print('test ', model.predict(text_np_array))
    prediction = model.predict_classes(text_np_array)[0][0]
    
    return  prediction, score

def get_article(url):
    article = Article(url, language='fr')
    article.download()
    article.parse()    

    return article
    
if __name__ == "__main__":
    app.run(debug=False)