from mlxtend.plotting import plot_decision_regions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from pandas.plotting import scatter_matrix
from pandas.plotting import andrews_curves
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import pickle

sns.set()
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class Model:

    def __init__(self): # param csv_name
        self.dataset = pd.read_csv('diabetes.csv')
        # self.dataset = pd.read_csv(csv_name)
        self.dataset_nonul = self.dataset  # copy of the data set

        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.K = None
        self.model = None

        self.process_data()
        self.scale_data()

        self.train()
        # self.model = pickle.load(open('model.pkl', 'rb'))

    def process_data(self):
        # change the 0 values that dont make sense with NAN
        self.dataset_nonul = self.dataset.copy(deep=True)
        self.dataset_nonul[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = self.dataset_nonul[
            ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NAN)

        # normally distributed data is replaced with the mean
        # skewed distributed data is replaced with the median
        self.dataset_nonul['Glucose'].fillna(self.dataset_nonul['Glucose'].mean(), inplace=True)
        self.dataset_nonul['BloodPressure'].fillna(self.dataset_nonul['BloodPressure'].mean(), inplace=True)
        self.dataset_nonul['SkinThickness'].fillna(self.dataset_nonul['SkinThickness'].median(), inplace=True)
        self.dataset_nonul['Insulin'].fillna(self.dataset_nonul['Insulin'].median(), inplace=True)
        self.dataset_nonul['BMI'].fillna(self.dataset_nonul['BMI'].mean(), inplace=True)

    def plot_data(self):
        # relationship between values and output
        scatter_matrix(self.dataset, figsize=(25, 25))
        sns.pairplot(self.dataset_nonul, hue='Outcome')

        # heatmap of value correlations - data with 0s
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.dataset.corr(), annot=True, cmap='RdYlGn')

        # heatmap of value correlations - data without 0s
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.dataset_nonul.corr(), annot=True, cmap='RdYlGn')

        plt.show()

    def scale_data(self):
        sc_X = StandardScaler()
        # X = self.dataset_nonul.drop("Outcome", axis=1)
        self.X = pd.DataFrame(sc_X.fit_transform(self.dataset_nonul.drop(["Outcome"], axis=1), ),
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                                       'BMI', 'DiabetesPedigreeFunction', 'Age'])
        self.y = self.dataset_nonul.Outcome
        # split training and test data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=1 / 3,
                                                                                random_state=42, stratify=self.y)

    def train(self):
        test_scores = []
        train_scores = []

        error = []

        for i in range(1, 50):
            knn = KNeighborsClassifier(i)
            train = knn.fit(self.X_train, self.y_train)
            predict = knn.predict(self.X_test)

            train_scores.append(knn.score(self.X_train, self.y_train))
            test_scores.append(knn.score(self.X_test, self.y_test))

            er = f1_score(predict, self.y_test)
            error.append(1 - er)
            print(
                "Accuracy of the model with k = {} is -> {} %".format(i, metrics.accuracy_score(self.y_test, predict)))

        # score that comes from testing on the same datapoints that were used for training
        max_train_score = max(train_scores)
        train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
        print('Max train score {} % and k = {}'.format(max_train_score * 100,
                                                       list(map(lambda x: x + 1, train_scores_ind))))

        # score that comes from testing on the datapoints that were split in the beginning to be used for testing solely
        max_test_score = max(test_scores)
        test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
        print(
            'Max test score {} % and k = {}'.format(max_test_score * 100, list(map(lambda x: x + 1, test_scores_ind))))

        # error that come for observing the elbow curve
        min_error = min(error)
        min_error_ind = error.index(min_error)
        print('Min error score {} % and k = {}'.format(min_error * 100, min_error_ind))
        error = list(map(lambda x: 1 - x, error))

        plt.figure(figsize=(12, 5))
        sns.lineplot(range(1, 50), train_scores, marker='*', label='Train Score')
        sns.lineplot(range(1, 50), test_scores, marker='o', label='Test Score')
        sns.lineplot(range(1, 50), error, marker='x', label='1 - Error')
        plt.xlabel("K")
        plt.show()

        self.K = test_scores_ind

        self.model = KNeighborsClassifier(self.K)
        self.model.fit(self.X_train, self.y_train)

        pickle.dump(self.model, open('model.pkl', 'wb'))
        self.model = pickle.load(open('model.pkl', 'rb'))

    def plot_model(self):
        print(self.model.score(self.X_test, self.y_test))

        value = 20000
        width = 20000
        plot_decision_regions(self.X.values, self.y.values, clf=self.model, legend=2,
                              filler_feature_values={2: value, 3: value, 4: value, 5: value, 6: value, 7: value},
                              filler_feature_ranges={2: width, 3: width, 4: width, 5: width, 6: width, 7: width},
                              X_highlight=self.X_test.values,
                              colors='red,green')

        # Adding axes annotations
        plt.title('KNN with Diabetes Data')
        plt.show()

        y_pred = self.model.predict(self.X_test)
        confusion_matrix(self.y_test, y_pred)
        conf_matrix = pd.crosstab(self.y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
        print(conf_matrix)

        cnf_matrix = metrics.confusion_matrix(self.y_test, y_pred)
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
        plt.title('Confusion matrix', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()

        plt.figure(figsize=(15, 10))
        sample = self.dataset_nonul.sample(frac=0.05)
        andrews_curves(sample, "Outcome", color=('red', 'yellow'))
        plt.title('Andrews Curves Plot', fontsize=20, fontweight='bold')
        plt.legend(loc=1, prop={'size': 15}, frameon=True, shadow=True, facecolor="white", edgecolor="black")
        plt.show()

        print(classification_report(self.y_test, y_pred))

    def plot_roc(self):
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)

        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='Knn')
        plt.xlabel('fpr')
        plt.ylabel('tpr')
        plt.title('Knn(n_neighbors = {}) ROC curve'.format(self.K))
        plt.show()

        # Area under ROC curve
        roc_auc_score(self.y_test, y_pred_proba)

        # In case of classifier like knn the parameter to be tuned is n_neighbors
        param_grid = {'n_neighbors': np.arange(1, 100)}
        knn = KNeighborsClassifier()
        knn_cv = GridSearchCV(knn, param_grid, cv=5)
        knn_cv.fit(self.X, self.y)

        print("Best Score:" + str(knn_cv.best_score_))
        print("Best Parameters: " + str(knn_cv.best_params_))

    def predict(self, data):
        self.model = pickle.load(open('model.pkl', 'rb'))
        output = self.model.predict(data)
        print(data, output)
        pickle.dump(self.model, open('model.pkl', 'wb'))
        return output


# print(dataset.head())
# print(dataset.info(verbose=True))
# print(dataset.describe(percentiles=[0.1, 0.2, 0.3, 0.6, 0.8]))
# print(dataset.describe().T)
#
# print("-----------------------------------")
# print(dataset.mean())
# print(dataset.median())
# print("-----------------------------------")

# # change the 0 values that dont make sense
# dataset_nonul = dataset.copy(deep=True)
# dataset_nonul[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = dataset_nonul[
#     ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NAN)
# print(dataset_nonul.isnull().sum())

# understanding the new null values found
# p = dataset.hist(figsize=(20, 20))
# plt.show()

# dataset_nonul['Glucose'].fillna(dataset_nonul['Glucose'].mean(), inplace=True)
# dataset_nonul['BloodPressure'].fillna(dataset_nonul['BloodPressure'].mean(), inplace=True)
# dataset_nonul['SkinThickness'].fillna(dataset_nonul['SkinThickness'].median(), inplace=True)
# dataset_nonul['Insulin'].fillna(dataset_nonul['Insulin'].median(), inplace=True)
# dataset_nonul['BMI'].fillna(dataset_nonul['BMI'].mean(), inplace=True)

# p = dataset_nonul.hist(figsize=(20, 20))
# plt.show()
#
# print(dataset.shape)

# age_group = sorted(set(dataset["Age"]))
# print(age_group)
#
# for age in age_group:
#     data_age = dataset_nonul[dataset_nonul["Age"] == age]
#     plt.scatter(data_age["BMI"], data_age["SkinThickness"])
#     plt.title(age)
#     plt.xlabel("BMI")
#     plt.ylabel("SkinThickness")
#     plt.show()

# for i in range(0, 2):
#     data_outcome = dataset_nonul[dataset_nonul["Outcome"] == i]
#     plt.scatter(data_outcome["Insulin"], data_outcome["Glucose"], 5)
#     plt.title("Diabetes result " + str(i))
#     plt.xlabel("Insulin level")
#     plt.ylabel("Glucose level")
#     plt.show()
#

# p = msno.bar(dataset)
# plt.show()

# color_wheel = {1: "#0392cf", 2: "#7bc043"}
# colors = dataset["Outcome"].map(lambda x: color_wheel.get(x + 1))
# print(dataset.Outcome.value_counts())
# p = dataset.Outcome.value_counts().plot(kind="bar")
# p.Color = colors
# plt.show()

# from pandas.plotting import scatter_matrix

#
# p = scatter_matrix(dataset, figsize=(25, 25))
# p = sns.pairplot(dataset_nonul, hue='Outcome')
#
# plt.figure(figsize=(12, 10))  # on this line I just set the size of figure to 12 by 10.
# p = sns.heatmap(dataset.corr(), annot=True, cmap='RdYlGn')  # seaborn has very simple solution for heatmap
#
# plt.figure(figsize=(12, 10))  # on this line I just set the size of figure to 12 by 10.
# p = sns.heatmap(dataset_nonul.corr(), annot=True, cmap='RdYlGn')  # seaborn has very simple solution for heatmap
#
# plt.show()

# from sklearn.preprocessing import StandardScaler
#
# sc_X = StandardScaler()
# X = pd.DataFrame(sc_X.fit_transform(dataset_nonul.drop(["Outcome"], axis=1), ),
#                  columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
#                           'BMI', 'DiabetesPedigreeFunction', 'Age'])
#
# print(X.head())
#
# # X = dataset_nonul.drop("Outcome", axis=1)
# y = dataset_nonul.Outcome

# importing train_test_split
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=42, stratify=y)
#
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import f1_score
# from sklearn import metrics

# test_scores = []
# train_scores = []
#
# error = []
#
# for i in range(1, 50):
#     knn = KNeighborsClassifier(i)
#     train = knn.fit(X_train, y_train)
#     predict = knn.predict(X_test)
#
#     train_scores.append(knn.score(X_train, y_train))
#     test_scores.append(knn.score(X_test, y_test))
#
#     er = f1_score(predict, y_test)
#     error.append(1 - er)
#     print("Accuracy of the model with k = {} is -> {} %".format(i, metrics.accuracy_score(y_test, predict)))
#
# # score that comes from testing on the same datapoints that were used for training
# max_train_score = max(train_scores)
# train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
# print('Max train score {} % and k = {}'.format(max_train_score * 100, list(map(lambda x: x + 1, train_scores_ind))))
#
# # score that comes from testing on the datapoints that were split in the beginning to be used for testing solely
# max_test_score = max(test_scores)
# test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
# print('Max test score {} % and k = {}'.format(max_test_score * 100, list(map(lambda x: x + 1, test_scores_ind))))
#
# # error that come for observing the elbow curve
# min_error = min(error)
# min_error_ind = error.index(min_error)
# print('Min error score {} % and k = {}'.format(min_error * 100, min_error_ind))
# error = list(map(lambda x: 1 - x, error))
#
# plt.figure(figsize=(12, 5))
# p = sns.lineplot(range(1, 50), train_scores, marker='*', label='Train Score')
# p = sns.lineplot(range(1, 50), test_scores, marker='o', label='Test Score')
# p = sns.lineplot(range(1, 50), error, marker='x', label='1 - Error')
#
# plt.xlabel("K")
#
# plt.show()

# knn = KNeighborsClassifier(11)
# # knn = KNeighborsClassifier(35)
# # knn = KNeighborsClassifier(28)
#
# knn.fit(self.X_train, self.y_train)
# print(knn.score(self.X_test, self.y_test))
#
# value = 20000
# width = 20000
# plot_decision_regions(self.X.values, self.y.values, clf=knn, legend=2,
#                       filler_feature_values={2: value, 3: value, 4: value, 5: value, 6: value, 7: value},
#                       filler_feature_ranges={2: width, 3: width, 4: width, 5: width, 6: width, 7: width},
#                       X_highlight=self.X_test.values,
#                       colors='red,green')
#
# # Adding axes annotations
# plt.title('KNN with Diabetes Data')
# plt.show()

# import confusion_matrix
# from sklearn.metrics import confusion_matrix

# let us get the predictions using the classifier we had fit above
# y_pred = knn.predict(self.X_test)
# confusion_matrix(self.y_test, y_pred)
# conf_matrx = pd.crosstab(self.y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
# print(conf_matrx)
#
# # from sklearn import metrics
#
# cnf_matrix = metrics.confusion_matrix(self.y_test, y_pred)
# p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
# plt.title('Confusion matrix', y=1.1)
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label')
# plt.show()
#
# # import classification_report
# # from sklearn.metrics import classification_report
#
# print(classification_report(y_test, y_pred))

# from sklearn.metrics import roc_curve

# y_pred_proba = knn.predict_proba(X_test)[:, 1]
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
#
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr, tpr, label='Knn')
# plt.xlabel('fpr')
# plt.ylabel('tpr')
# plt.title('Knn(n_neighbors=11) ROC curve')
# plt.show()
#
# # Area under ROC curve
# # from sklearn.metrics import roc_auc_score
#
# roc_auc_score(y_test, y_pred_proba)
#
# # import GridSearchCV
# # from sklearn.model_selection import GridSearchCV
#
# # In case of classifier like knn the parameter to be tuned is n_neighbors
# param_grid = {'n_neighbors': np.arange(1, 100)}
# knn = KNeighborsClassifier()
# knn_cv = GridSearchCV(knn, param_grid, cv=5)
# knn_cv.fit(X, y)
#
# print("Best Score:" + str(knn_cv.best_score_))
# print("Best Parameters: " + str(knn_cv.best_params_))