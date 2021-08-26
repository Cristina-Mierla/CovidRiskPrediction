import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
from pandas.plotting import andrews_curves
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
# from xgboost import XGBClassifier

from DataProcessing import DataProcessing

sns.set()
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class Model:

    def __init__(self):  # param csv_name
        print("\n\t\tTraining and Testing Prediction Models\n")
        self.csv_name = 'dataset.csv'
        data_process = DataProcessing(self.csv_name)
        self.dataset = data_process.get_dataset()
        # self.dataset = pd.read_csv(csv_name)
        self.scaled_dataset = self.dataset  # copy of the data set

        self.y = pd.DataFrame()
        self.X = pd.DataFrame()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.K = None
        self.model = None

        self.accuracy = 0

        # self.scale_data()

        self.train()

        # self.plot_data()
        # self.model = pickle.load(open('model.pkl', 'rb'))

        data_process.plt_data_distribution()
        data_process.plt_stare_externare()

    def scale_data(self):
        sc_X = StandardScaler()
        print(self.scaled_dataset)
        self.X = pd.DataFrame(
            sc_X.fit_transform(self.scaled_dataset.drop(["stare_externare"], axis=1), ))
        self.y = self.scaled_dataset[["stare_externare"]]

        # split training and test data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=1 / 3,
                                                                                random_state=56)

        self.X_train = sc_X.fit_transform(self.X_train)
        self.X_test = sc_X.transform(self.X_test)

    def train(self):
        self.scale_data()

        # KfValidation = KFold(n_splits=4, shuffle=False)
        # Kf = KfValidation.split(self.X)

        max_error_scoring = 'max_error'
        neg_mae_scoring = 'neg_mean_absolute_error'
        r2_scoring = 'r2'
        neg_mse_scoring = 'neg_mean_squared_error'

        dfs = []

        # models = [('LR', LinearRegression()), ('LASSO', Lasso()), ('EN', ElasticNet()), ('RIDGE', Ridge()),
        #           ('KNN', KNeighborsRegressor()), ('CART', DecisionTreeRegressor()), ('SVR', SVR()),
        #           ('GAUSS', GaussianProcessClassifier()), ('TREECLASS', DecisionTreeClassifier()),
        #           ('FOREST', RandomForestClassifier()), ('BAYES', GaussianNB()), ('ADA', AdaBoostClassifier()),
        #           ('SVC', SVC()), ('NEURONAL', MLPClassifier())]
        models = [('LOGREG', LogisticRegression()), ('RF', RandomForestClassifier()), ('KNN', KNeighborsClassifier()),
                  ('SVM', SVC()), ('TREECLASS', DecisionTreeClassifier()), ('ADA', AdaBoostClassifier()),
                  ('GNB', GaussianProcessClassifier()), ('MLP', MLPClassifier())]
        # , ('CART', DecisionTreeRegressor()), ('TREECLASS', DecisionTreeClassifier()),
        # models = [('LOGREG', LogisticRegression()), ('RF', RandomForestClassifier()), ('SVM', SVC()), ('KNN', KNeighborsClassifier())]
        # training and testing the above models
        results = []
        names = []
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        # target_names = ['Usor', 'Moderat', 'Sever'] # forma boala
        # target_names = ['Vindecat', 'Ameliorat', 'Stationar', 'Decedat']
        target_names = ['Vindecat', 'Ameliorat', 'Stationar', 'Agravat', 'Decedat'] # stare externare

        # for name, model in models:
        #     kfold = KFold(n_splits=10, random_state=7, shuffle=True)
        #
        #     # res1 = cross_val_score(model, self.X, self.y, cv=kfold, scoring=max_error_scoring)
        #     res2 = cross_val_score(model, self.X, self.y, cv=kfold, scoring=neg_mae_scoring)
        #     res3 = cross_val_score(model, self.X, self.y, cv=kfold, scoring=r2_scoring)
        #     res4 = cross_val_score(model, self.X, self.y, cv=kfold, scoring=neg_mse_scoring)
        #
        #     msg = '%6s -> mean absolute error: %.4f, r2: %.4f, mean squared error: %.4f' % (
        #         name, -res2.mean(), res3.mean(), -res4.mean())
        #     print(msg)

        for name, model in models:
            kfold = KFold(n_splits=5, shuffle=True, random_state=90210)
            cv_results = cross_validate(model, self.X_train, self.y_train, cv=kfold, scoring=scoring)
            clf = model.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)
            print(name)
            print(classification_report(self.y_test, y_pred, target_names=target_names))

            results.append(cv_results)
            names.append(name)

            this_df = pd.DataFrame(cv_results)
            this_df['model'] = name
            dfs.append(this_df)

        final = pd.concat(dfs, ignore_index=True)
        print(final)

        test_scores = []
        train_scores = []

        error = []

        # for i in range(1, 50):
        #     knn = KNeighborsClassifier(i)
        #     train = knn.fit(self.X_train, self.y_train)
        #     predict = knn.predict(self.X_test)
        #
        #     train_scores.append(knn.score(self.X_train, self.y_train))
        #     test_scores.append(knn.score(self.X_test, self.y_test))
        #
        #     er = f1_score(predict, self.y_test)
        #     error.append(1 - er)
        #     print(
        #         "Accuracy of the model with k = {} is -> {} %".format(i, metrics.accuracy_score(self.y_test, predict)))
        #
        # # score that comes from testing on the same data points that were used for training
        # max_train_score = max(train_scores)
        # train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
        # print('Max train score {} % and k = {}'.format(max_train_score * 100,
        #                                                list(map(lambda x: x + 1, train_scores_ind))))
        #
        # # score that comes from testing on the data points that were split in the beginning to be used for testing solely
        # max_test_score = max(test_scores)
        # test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
        # print(
        #     'Max test score {} % and k = {}'.format(max_test_score * 100, list(map(lambda x: x + 1, test_scores_ind))))
        #
        # # error that come for observing the elbow curve
        # min_error = min(error)
        # min_error_ind = error.index(min_error)
        # print('Min error score {} % and k = {}'.format(min_error * 100, min_error_ind))
        # error = list(map(lambda x: 1 - x, error))
        #
        # plt.figure(figsize=(12, 5))
        # sns.lineplot(range(1, 50), train_scores, marker='*', label='Train Score')
        # sns.lineplot(range(1, 50), test_scores, marker='o', label='Test Score')
        # sns.lineplot(range(1, 50), error, marker='x', label='1 - Error')
        # plt.xlim([5, 25])
        # plt.xlabel("K")
        # plt.show()
        #
        # self.K = list(map(lambda x: x + 1, test_scores_ind))[0]
        # self.accuracy = max_test_score
        #
        # self.model = KNeighborsClassifier(self.K)
        # self.model.fit(self.X_train, self.y_train)
        #
        # pickle.dump(self.model, open('model.pkl', 'wb'))
        # self.model = pickle.load(open('model.pkl', 'rb'))

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
        sample = self.scaled_dataset.sample(frac=0.05)
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
        # print("Prediction accuracy -> {} %".format(self.accuracy * 100))
        self.model = pickle.load(open('model.pkl', 'rb'))
        output = self.model.predict(data)
        print(data, output)
        data = data[0]
        record = {
            "Pregnancies": [data[0]],
            "Glucose": [data[1]],
            "BloodPressure": [data[2]],
            "SkinThickness": [data[3]],
            "Insulin": [data[4]],
            "BMI": [data[5]],
            "DiabetesPedigreeFunction": [data[6]],
            "Age": [data[7]],
            "Outcome": [output[0]]
        }
        df = pd.DataFrame(record)
        # df.to_csv('dataset_cop.csv', mode='a', index=False, header=False)
        pickle.dump(self.model, open('model.pkl', 'wb'))
        return output
