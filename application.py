from flask import Flask, request, render_template
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

application = Flask(__name__)

app = application

# route for a home package
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
                 gender = request.form.get("gender"),
                 race_ethnicity = request.form.get("ethnicity"),
                 parental_level_of_education = request.form.get("parental_level_of_education"),
                 lunch = request.form.get("lunch"),
                 test_preparation_course = request.form.get("test_preparation_course"),
                 reading_score = float(request.form.get("reading_score")),
                 writing_score = float(request.form.get("writing_score"))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        logging.info("before prediction")

        predict_pipeline = PredictPipeline()
        logging.info("mid prediction")

        results = predict_pipeline.predict(pred_df)
        logging.info("after prediction")

        return render_template("home.html", results=results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0")