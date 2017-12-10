from flask import Flask
import numpy
from flask import json
from flask import abort
from flask import request
import os

app = Flask(__name__)
@app.route('/')
def test():
  return "hello world"

@app.route('/predict', methods=['GET', 'POST'])
def predict_house_price():
  # check and parse input json
  try:
    input1 = request.json
    crime_rate = float(input1['crime_rate'])
    avg_number_of_rooms = float(input1['avg_number_of_rooms'])
    distance_to_employment_centers = float(input1['distance_to_employment_centers'])
    property_tax_rate = float(input1['property_tax_rate'])
    pupil_teacher_ratio = float(input1['pupil_teacher_ratio'])
    crime_rate = 1.0   
  except:
    abort(400)
  
  intercept = 1.0
  data = [crime_rate, avg_number_of_rooms, distance_to_employment_centers, property_tax_rate, pupil_teacher_ratio, intercept]
  
  # read model
  #filename = "/home/ekhnaton/Documents/Inka/housing_model.txt"
  #filename = "/home/muti/Documents/housing_model.txt"
  filename = '/app/housing_model.txt'
  model=numpy.genfromtxt(filename,delimiter=',',dtype=float,skip_header=0)
  
  # compute prediction
  #prediction = data*model
  
  # format as json
  prediction = {}
  prediction['house_value'] = sum(data*model)
  #prediction['stddev'] = ??
  prediction_json = json.dumps(prediction)
  
  
  return prediction_json

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0',port=port)


