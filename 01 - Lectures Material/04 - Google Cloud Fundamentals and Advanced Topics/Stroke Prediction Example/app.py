import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)


# Load the trained pipeline
pipeline_filename = os.path.join('trained_model', 'stroke_prediction_pipeline.pkl')
pipeline = joblib.load(pipeline_filename)

@app.route("/")
def home():
    return jsonify({
        "Message": "app up and running successfully"
    })

@app.route('/forecast', methods=['GET','POST'])
def forecast():
    """
    Predict stroke probability for new data
    ---
    parameters:
      - name: age
        in: formData
        type: number
        required: true
      - name: hypertension
        in: formData
        type: number
        required: true
      - name: heart_disease
        in: formData
        type: number
        required: true
      - name: avg_glucose_level
        in: formData
        type: number
        required: true
      - name: bmi
        in: formData
        type: number
        required: true
      - name: gender
        in: formData
        type: string
        required: true
      - name: ever_married
        in: formData
        type: string
        required: true
      - name: work_type
        in: formData
        type: string
        required: true
      - name: Residence_type
        in: formData
        type: string
        required: true
      - name: smoking_status
        in: formData
        type: string
        required: true
    responses:
      200:
        description: Prediction result is Stoke or No Stroke
    """
    if request.method == 'POST':
        data = request.form.to_dict()
        new_data = pd.DataFrame(data, index=[0])
        prediction = pipeline.predict(new_data)
        pred_mapping = {0: 'No-Stroke', 1: 'Stroke'}
        predicted_class = pred_mapping[prediction[0]]
        return jsonify({"Model prediction is": predicted_class})
    return '''
        <form method="POST">

            Age: <input type="number" name="age"><br>
            Hypertension (0 for No, 1 for Yes): <input type="number" name="hypertension"><br>
            Heart Disease (0 for No, 1 for Yes): <input type="number" name="heart_disease"><br>
            Average Glucose Level: <input type="number" name="avg_glucose_level"><br>
            Body Mass Index (BMI): <input type="number" name="bmi"><br>
            Fender (Male, Female, Other): <input type="string" name="gender"><br>
            Marital Status (Yes, No): <input type="string" name="ever_married"><br>
            Type of Work (Private, Self-Employed etc.): <input type="string" name="work_type"><br>
            Residence Type (Urban, Rural): <input type="string" name="Residence_type"><br>
            Smoking Status (formerly smoked, never smoked etc.): <input type="string" name="smoking_status"><br>

            <input type="submit" value="Submit">
        </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
