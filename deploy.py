from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
  # Get user input from the form
  sepal_length = request.form['sepal_length']
  sepal_width = request.form['sepal_width']
  petal_length = request.form['petal_length']
  petal_width = request.form['petal_width']

  # Load your machine learning model (replace with your model loading logic)
  filename = 'saved_model.pkl'
  try:
    with open(filename, 'rb') as file:
      model = pickle.load(file)
  except Exception as e:
    print(f"Error loading the model: {e}")
    return "Error: Failed to load model"

  # Prepare the input data for the model
  data = [[float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]]

  # Make prediction using the model
  prediction = model.predict(data)[0]

  # Display the prediction on the webpage
  return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
  app.run(debug=True)
