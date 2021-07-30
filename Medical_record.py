import model
class Medical_record:

    def __init__(self, preg, gluc, blpr, skth, insu, bmi, pedi, age):
        self.__preg = preg
        self.__gluc = gluc
        self.__blpr = blpr
        self.__skth = skth
        self.__insu = insu
        self.__bmi = bmi
        self.__pedi = pedi
        self.__age = age

        self.data = [[self.__preg, self.__gluc, self.__blpr, self.__skth, self.__insu, self.__bmi, self.__pedi, self.__age]]

    def predict(self, prediction_model):
        output = prediction_model.predict(self.data)
        record = self.data[0]
        record.append(output[0])
        print(self.data)
        # save new data in the csv
        # return the result
        # print the result in a nice manner
        return output

'''
plan:
    create instance of prediction model at the beginning of the program
    -> data is cleaned and scaled
    -> model is trained or the existing model is loaded
    GET the JSON input
    initialize a Medical_record class with the given values
    call the predict function over the given data
    -> predict the output
    -> save the data in the csv
    -> return (POST) the outcome
    
    (-> return (POST) a visualization of the model or of the result)
'''