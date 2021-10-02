from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib

# Load the model
model = joblib.load(open("emotion_classifier_pipe_lr_3_sept_2021.pkl",'rb'))

app = Flask(__name__)

@app.route('/')
def main():
	return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def home():
	# Get the text from the form
	input_text = request.form['inputText']

	if (input_text == ""):
		pred = ''
	else:
		pred = model.predict([input_text])
		pred = pred[0]
		pred = pred.capitalize()
	return render_template('home.html', data = pred, textVal = input_text)

if __name__ == '__main__':
	app.run(debug = True)