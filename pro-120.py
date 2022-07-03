import pandas as pd
from sklearn import datasets
wine = datasets.load_wine()
from sklearn.naives_bayes import GaussiansNB
gnb = GaussiansNB()
gnb.fit(x_train,y_train)
y_predict = gnb.predict(x_test)
accuracy= accuracy_score(y_test,y_predict)
print(accuracy)