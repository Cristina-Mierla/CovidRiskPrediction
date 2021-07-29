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

        self.data = [self.__preg, self.__gluc, self.__blpr, self.__skth, self.__insu, self.__bmi, self.__pedi, self.__age]

    def predict(self):
        output = model.predict(self.data)
        # save new data in the csv
        # return the result
        # print the result in a nice manner