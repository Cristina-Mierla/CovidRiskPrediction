from mlxtend.plotting import plot_decision_regions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

dataset = pd.read_csv('diabetes.csv')
print(dataset.head())
print(dataset.info(verbose=True))
print(dataset.describe(percentiles=[0.1, 0.2, 0.3, 0.6, 0.8]))
print(dataset.describe().T)
print("-----------------------------------")
print(dataset.mean())
print(dataset.median())
print("-----------------------------------")

# change the 0 values that dont make sense
dataset_nonul = dataset.copy(deep=True)
dataset_nonul[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = dataset_nonul[
    ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NAN)
print(dataset_nonul.isnull().sum())

# understanding the new null values found
# p = dataset.hist(figsize=(20, 20))
# plt.show()

dataset_nonul['Glucose'].fillna(dataset_nonul['Glucose'].mean(), inplace=True)
dataset_nonul['BloodPressure'].fillna(dataset_nonul['BloodPressure'].mean(), inplace=True)
dataset_nonul['SkinThickness'].fillna(dataset_nonul['SkinThickness'].median(), inplace=True)
dataset_nonul['Insulin'].fillna(dataset_nonul['Insulin'].median(), inplace=True)
dataset_nonul['BMI'].fillna(dataset_nonul['BMI'].mean(), inplace=True)

# p = dataset_nonul.hist(figsize=(20, 20))
# plt.show()

print(dataset.shape)

import missingno as msno
# p = msno.bar(dataset)
# plt.show()
#
# color_wheel = {1: "#0392cf", 2: "#7bc043"}
# colors = dataset["Outcome"].map(lambda x: color_wheel.get(x + 1))
# print(dataset.Outcome.value_counts())
# p = dataset.Outcome.value_counts().plot(kind="bar")
# p.Color = colors
# plt.show()

from pandas.plotting import scatter_matrix

# p = scatter_matrix(dataset, figsize=(25, 25))
# p = sns.pairplot(dataset_nonul, hue='Outcome')
#
# plt.figure(figsize=(12, 10))  # on this line I just set the size of figure to 12 by 10.
# p = sns.heatmap(dataset.corr(), annot=True, cmap='RdYlGn')  # seaborn has very simple solution for heatmap
#
# plt.figure(figsize=(12, 10))  # on this line I just set the size of figure to 12 by 10.
# p = sns.heatmap(dataset_nonul.corr(), annot=True, cmap='RdYlGn')  # seaborn has very simple solution for heatmap

# plt.show()

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(dataset_nonul.drop(["Outcome"], axis=1), ),
                 columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                          'BMI', 'DiabetesPedigreeFunction', 'Age'])

print(X.head())

# X = dataset_nonul.drop("Outcome", axis=1)
y = dataset_nonul.Outcome

# importing train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42, stratify=y)

from sklearn.neighbors import KNeighborsClassifier

# test_scores = []
# train_scores = []
#
# for i in range(1, 15):
#     knn = KNeighborsClassifier(i)
#     knn.fit(X_train, y_train)
#
#     train_scores.append(knn.score(X_train, y_train))
#     test_scores.append(knn.score(X_test, y_test))
#
# # score that comes from testing on the same datapoints that were used for training
# max_train_score = max(train_scores)
# train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
# print('Max train score {} % and k = {}'.format(max_train_score*100, list(map(lambda x: x+1, train_scores_ind))))
#
# # score that comes from testing on the datapoints that were split in the beginning to be used for testing solely
# max_test_score = max(test_scores)
# test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
# print('Max test score {} % and k = {}'.format(max_test_score*100, list(map(lambda x: x+1, test_scores_ind))))
#
# plt.figure(figsize=(12,5))
# p = sns.lineplot(range(1,15),train_scores,marker='*', label='Train Score')
# p = sns.lineplot(range(1,15),test_scores,marker='o', label='Test Score')

# plt.show()

knn = KNeighborsClassifier(11)
# knn = KNeighborsClassifier(35)
# knn = KNeighborsClassifier(28)

knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))

value = 20000
width = 20000
plot_decision_regions(X.values, y.values, clf=knn, legend=2,
                      filler_feature_values={2: value, 3: value, 4: value, 5: value, 6: value, 7: value},
                      filler_feature_ranges={2: width, 3: width, 4: width, 5: width, 6: width, 7: width},
                      X_highlight=X_test.values,
                      colors='red,green')

# Adding axes annotations
plt.title('KNN with Diabetes Data')
plt.show()


# import confusion_matrix
from sklearn.metrics import confusion_matrix
# let us get the predictions using the classifier we had fit above
y_pred = knn.predict(X_test)
confusion_matrix(y_test, y_pred)
conf_matrx = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
print(conf_matrx)

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# import classification_report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

from sklearn.metrics import roc_curve
y_pred_proba = knn.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=11) ROC curve')
plt.show()

# Area under ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_proba)

# import GridSearchCV
from sklearn.model_selection import GridSearchCV
# In case of classifier like knn the parameter to be tuned is n_neighbors
param_grid = {'n_neighbors': np.arange(1, 100)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X, y)

print("Best Score:" + str(knn_cv.best_score_))
print("Best Parameters: " + str(knn_cv.best_params_))
