import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SMOTENC, ADASYN
from imblearn.under_sampling import RandomUnderSampler
# from mlxtend.plotting import plot_decision_regions
from pandas.plotting import andrews_curves
from sklearn import metrics
from sklearn import tree
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
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
        self.dataset = pd.read_csv(self.csv_name, parse_dates=True)
        self.data_process = DataProcessing(self.dataset)
        self.dataset = self.data_process.get_dataset()
        # self.dataset = pd.read_csv(csv_name)
        self.scaled_dataset = self.dataset  # copy of the data set

        self.y = pd.DataFrame()
        self.X = pd.DataFrame()
        self.target_names = None
        self.X_train = None
        self.X_test = None
        self.X_valid = None
        self.y_train = None
        self.y_test = None
        self.y_valid = None

        self.K = None
        self.model = None

        self.accuracy = 0

        # self.test_models()

        self.train()

        self.pre_pruning()

        self.plot_model()
        # self.model = pickle.load(open('model.pkl', 'rb'))

        # data_process.plt_data_distribution()
        # data_process.plt_stare_externare()

    def scale_data(self):
        sc_X = StandardScaler()
        print(self.scaled_dataset)
        self.X = pd.DataFrame(
            sc_X.fit_transform(self.scaled_dataset.drop(["stare_externare", "forma_boala"], axis=1), ))
        self.y = self.scaled_dataset[["stare_externare"]]

        oversample = ADASYN()
        self.X, self.y = oversample.fit_resample(self.X, self.y)
        #
        # rus = RandomUnderSampler(random_state=0)
        # rus.fit(self.X, self.y)

        # split training and test data
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=1 / 3,
        #                                                                         random_state=56)

        self.X_train, X_rem, self.y_train, y_rem = train_test_split(self.X, self.y, train_size=0.6,
                                                                    random_state=56)

        self.X_valid, self.X_test, self.y_valid, self.y_test = train_test_split(X_rem, y_rem, test_size=1 / 2,
                                                                                random_state=56)

        self.X_train = sc_X.fit_transform(self.X_train)
        self.X_test = sc_X.transform(self.X_test)
        self.X_valid = sc_X.transform(self.X_valid)

    def test_models(self):
        self.scale_data()

        max_error_scoring = 'max_error'
        neg_mae_scoring = 'neg_mean_absolute_error'
        r2_scoring = 'r2'
        neg_mse_scoring = 'neg_mean_squared_error'

        dfs = []

        models = [('LOGREG', LogisticRegression()), ('RF', RandomForestClassifier()), ('KNN', KNeighborsClassifier()),
                  ('SVM', SVC()), ('TREECLASS', DecisionTreeClassifier()), ('ADA', AdaBoostClassifier()),
                  ('GNB', GaussianProcessClassifier()), ('MLP', MLPClassifier())]

        # training and testing the above models
        results = []
        names = []
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

        # target_names = ['Usor', 'Moderat', 'Sever'] # forma boala
        # target_names = ['Vindecat', 'Ameliorat', 'Stationar', 'Decedat']
        target_names = ['Vindecat', 'Ameliorat', 'Stationar', 'Agravat', 'Decedat']  # stare externare

        for name, model in models:
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=90210)
            cv_results = cross_validate(model, self.X_train, self.y_train, cv=kfold, scoring=scoring)
            clf = model.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_valid)
            print(name)
            print(classification_report(self.y_valid, y_pred, target_names=target_names))

            results.append(cv_results)
            names.append(name)

            this_df = pd.DataFrame(cv_results)
            this_df['model'] = name
            dfs.append(this_df)

        final = pd.concat(dfs, ignore_index=True)

    def train(self):
        self.scale_data()

        self.target_names = ['Vindecat', 'Ameliorat', 'Stationar', 'Agravat', 'Decedat']  # stare externare

        max_error_scoring = 'max_error'
        neg_mae_scoring = 'neg_mean_absolute_error'
        r2_scoring = 'r2'
        neg_mse_scoring = 'neg_mean_squared_error'

        results = []
        names = []
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

        self.model = DecisionTreeClassifier()
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=90210)
        cv_results = cross_validate(self.model, self.X_train, self.y_train, cv=kfold, scoring=scoring)
        clf = self.model.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        print("Decision Tree")
        print(classification_report(self.y_test, y_pred, target_names=self.target_names))

        # self.model = KNeighborsClassifier(self.K)
        # self.model.fit(self.X_train, self.y_train)
        #
        # pickle.dump(self.model, open('model.pkl', 'wb'))
        # self.model = pickle.load(open('model.pkl', 'rb'))

    def pre_pruning(self):
        params = {'max_depth': [2, 4, 6, 8, 10, 12],
                  'min_samples_split': [2, 3, 4],
                  'min_samples_leaf': [1, 2]}

        gcv = GridSearchCV(estimator=self.model, param_grid=params)
        gcv.fit(self.X_train, self.y_train)

        self.model = gcv.best_estimator_
        self.model.fit(self.X_train, self.y_train)

    def plot_confusionmatrix(self, y_train_pred, y_train, dom):
        print(f'{dom} Confusion matrix')
        cf = confusion_matrix(y_train_pred, y_train)
        sns.heatmap(cf, annot=True, yticklabels=self.target_names
                    , xticklabels=self.target_names, cmap='Blues', fmt='g')
        plt.tight_layout()
        plt.show()

    def plot_model(self):
        print(self.model.score(self.X_test, self.y_test))

        features = self.X
        plt.figure(figsize=(55, 55))
        tree.plot_tree(self.model, feature_names=features, class_names=self.target_names, filled=True, fontsize=4)
        plt.show()

        enmax_palette = ["#e5833c", "#7fe63e", "#53e8cc", "#3c39e5", "#e539c0"]

        sns.set_palette(palette=enmax_palette)

        colors = pd.DataFrame({'Stare de externare': ['Vindecat', 'Ameliorat', 'Stationar', 'Agravat', 'Decedat'], 'val': [1, 1, 1, 1, 1]})
        fig = sns.barplot(x='val', y='Stare de externare', data=colors)
        fig.set_xticklabels([])
        fig.set_xlabel('')
        plt.show()

        value = 20000
        width = 20000
        # plot_decision_regions(self.X.values, self.y.values, clf=self.model, legend=2,
        #                       filler_feature_values={2: value, 3: value, 4: value, 5: value, 6: value, 7: value},
        #                       filler_feature_ranges={2: width, 3: width, 4: width, 5: width, 6: width, 7: width},
        #                       X_highlight=self.X_test.values,
        #                       colors='red,green')
        #
        # # Adding axes annotations
        # plt.title('KNN with Diabetes Data')
        # plt.show()

        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        # confusion_matrix(self.y_test, y_test_pred)
        # conf_matrix = pd.crosstab(self.y_test, y_test_pred, rownames=['True'], colnames=['Predicted'], margins=True)
        # print(conf_matrix)

        print(f'Train score {metrics.accuracy_score(y_train_pred, self.y_train)}')
        print(f'Test score {metrics.accuracy_score(y_test_pred, self.y_test)}')
        self.plot_confusionmatrix(self.y_train, y_train_pred, dom='Train')
        self.plot_confusionmatrix(self.y_test, y_test_pred,  dom='Test')
        plt.show()

        cnf_matrix = metrics.confusion_matrix(self.y_test, y_test_pred)
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="Blues", fmt='g')
        plt.title('Confusion matrix', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()

        plt.figure(figsize=(15, 10))
        sample = self.scaled_dataset.sample(frac=0.05)
        andrews_curves(sample, "stare_externare", color=('red', 'yellow'))
        plt.title('Andrews Curves Plot', fontsize=20, fontweight='bold')
        plt.legend(loc=1, prop={'size': 15}, frameon=True, shadow=True, facecolor="white", edgecolor="black")
        plt.show()

        print(classification_report(self.y_test, y_test_pred))

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

    def predict(self, prediction_data):
        # prediction_data = [age, sex, diagnos_int, spitalizare, ati, analize]
        # print("Prediction accuracy -> {} %".format(self.accuracy * 100))

        # self.model = pickle.load(open('model.pkl', 'rb'))

        prediction_set = self.data_process.predict(prediction_data)
        output1 = self.model.predict(prediction_set)
        print(output1)
        output2 = self.model.predict_proba(prediction_set)
        print(output2)

        # df.to_csv('dataset_cop.csv', mode='a', index=False, header=False)
        # pickle.dump(self.model, open('model.pkl', 'wb'))

        return output1, output2


