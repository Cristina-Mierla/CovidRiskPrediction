# import main Flask class and request object
import json

from flask import Flask, request
from model import Model

# create the Flask app
app = Flask(__name__)
prediction_model = Model()


@app.route('/medical-record', methods=['POST', 'GET'])
def get_med_rec():
    age = None
    diagnos_int = None
    sex = None

    # daca exista si e json
    if request:

        # request_data = request.get_json()

        # if 'Pregnancies' in request_data:
        #     pregnancies = request_data["Pregnancies"]
        #
        # if 'Glucose' in request_data:
        #     glucose = request_data["Glucose"]
        #
        # if 'BloodPressure' in request_data:
        #     bldpressure = request_data["BloodPressure"]
        #
        # if 'SkinThickness' in request_data:
        #     skinthick = request_data['SkinThickness']
        #
        # if 'Insulin' in request_data:
        #     insulin = request_data['Insulin']
        #
        # if 'BMI' in request_data:
        #     bmi = request_data['BMI']
        #
        # if 'DiabetesPedigreeFunction' in request_data:
        #     pedigree = request_data['DiabetesPedigreeFunction']
        #
        # if 'Age' in request_data:
        #     age = request_data['Age']

        form = request.form

        age = json.loads(form["RiskPred"])["Age"]
        sex = json.loads(form["RiskPred"])["Gender"]
        diagnos_int = json.loads(form["RiskPred"])["Admission_diagnostic"]

        # save the input



        # return "JSON was received and the prediction was -> {}".format(prediction[0]), 200
        if 1:
            return "The patient has a high risk of diabetes", 200
        return "The patient has a low risk of diabetes", 200

    else:

        return "Invalid request", 400


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
