# import main Flask class and request object
from flask import Flask, request
from Medical_record import Medical_record
from model import Model

# create the Flask app
app = Flask(__name__)
prediction_model = Model()


@app.route('/medical-record', methods=['GET'])
def get_med_rec():
    pregnancies = None
    glucose = None
    bldpressure = None
    skinthick = None
    insulin = None
    bmi = None
    pedigree = None
    age = None

    # daca exista si e json
    if request and request.is_json:

        request_data = request.get_json()

        if 'Pregnancies' in request_data:
            pregnancies = request_data["Pregnancies"]

        if 'Glucose' in request_data:
            glucose = request_data["Glucose"]

        if 'BloodPressure' in request_data:
            bldpressure = request_data["BloodPressure"]

        if 'SkinThickness' in request_data:
            skinthick = request_data['SkinThickness']

        if 'Insulin' in request_data:
            insulin = request_data['Insulin']

        if 'BMI' in request_data:
            bmi = request_data['BMI']

        if 'DiabetesPedigreeFunction' in request_data:
            pedigree = request_data['DiabetesPedigreeFunction']

        if 'Age' in request_data:
            age = request_data['Age']

        record = Medical_record(pregnancies, glucose, bldpressure, skinthick, insulin, bmi, pedigree, age)
        prediction = record.predict(prediction_model)

        return "JSON was received and the prediction was -> {}".format(prediction[0]), 200

    else:

        return "Invalid request", 400


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
