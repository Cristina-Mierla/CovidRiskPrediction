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
    spitalizare = None
    ati = None
    analize = None

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
        spitalizare = json.loads(form["RiskPred"])["Hospitalization"]
        ati = json.loads(form["RiskPred"])["ATI"]
        analize = json.loads(form['RiskPred'])['Analyzes']

        prediction_data = [age, sex, diagnos_int, spitalizare, ati, analize]

        prediction_result, prediction_percentage = prediction_model.predict(prediction_data)

        label = {0: 'Vindecat', 1: 'Ameliorat', 2: 'Stationar', 3: 'Agravat', 4: 'Decedat'}

        result = "The patient has a high chance to be release as: {}\n\n".format(label[prediction_result[0]])

        for i in range(0, 5):
            aux = "\t{:20} -> {}%\n".format(label[i], round(prediction_percentage[0][i]*100, 0))
            result = result + aux

        print(result)

        return result, 200

    else:

        return "Invalid request", 400


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
