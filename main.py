# import main Flask class and request object
import json

from flask import Flask, request, send_file
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
    id = None

    # daca exista si e json
    if request:

        # request_data = request.get_json()

        form = request.form

        age = json.loads(form["RiskPred"])["Age"]
        sex = json.loads(form["RiskPred"])["Gender"]
        diagnos_int = json.loads(form["RiskPred"])["Admission_diagnostic"]
        spitalizare = json.loads(form["RiskPred"])["Hospitalization"]
        ati = json.loads(form["RiskPred"])["ATI"]
        analize = json.loads(form['RiskPred'])['Analyzes']
        comorb = json.loads(form['RiskPred'])['Comorbidities']
        id = json.loads(form['RiskPred'])['Id']

        prediction_data = [age, sex, diagnos_int, spitalizare, ati, analize, comorb, id]

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


@app.route('/statistics', methods=['POST', 'GET'])
def get_stat():
    age = None
    diagnos_int = None
    sex = None
    spitalizare = None
    ati = None
    analize = None
    id = None

    # daca exista si e json
    if request:

        # request_data = request.get_json()

        form = request.form

        age = json.loads(form["RiskPred"])["Age"]
        sex = json.loads(form["RiskPred"])["Gender"]
        diagnos_int = json.loads(form["RiskPred"])["Admission_diagnostic"]
        spitalizare = json.loads(form["RiskPred"])["Hospitalization"]
        ati = json.loads(form["RiskPred"])["ATI"]
        analize = json.loads(form['RiskPred'])['Analyzes']
        comorb = json.loads(form['RiskPred'])['Comorbidities']
        id = json.loads(form['RiskPred'])['Id']

        prediction_data = [age, sex, diagnos_int, spitalizare, ati, analize, comorb, id]

        filename = prediction_model.statistics(prediction_data)

        return send_file(filename, mimetype='image/png'), 200

        # return result, 200

    else:

        return "Invalid request", 400


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
