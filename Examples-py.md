Supervised
```
import numpy as np
import pandas an pd 
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

dd = pd.read_csv("https://s3-us-west-2.amazonaws.com/researchs/car.csv")

X = pd.get_dummies(dd.drop("Class", axis = 1))
X["Class"] = dd.Class

idx = np.random.uniform(0, 1, len(X)) <= .8

cp = cpir_gel(source_data_ = X[idx == True].reset_index().drop('index', axis = 1),
              k = 10, learning_method = 'supervised', class_var = "Class")

clf = GaussianNB()
clf.fit(X = pd.DataFrame(cp[0]), y = pd.factorize(cp[2].Class)[0])
pred = clf.predict(pd.DataFrame(np.matmul(X.drop("Class", axis = 1)[idx == False], cp[1])))

pd.crosstab(X.Class[idx == False], pred, rownames=['Actual'], colnames=['Predicted'])
accuracy_score(pd.factorize(X.Class[idx == False])[0], pred)
```

Unsupervised

```
X = pd.get_dummies(dd.drop("Class", axis = 1))

cp_u = cpir_gel(source_data_ = X[idx == True].reset_index().drop('index', axis = 1), 
k = 10, learning_method = 'unsupervised')

clf = RandomForestClassifier()
clf.fit(X = pd.DataFrame(cp[0]), y = pd.factorize(dd.Class[idx == True])[0])
pred = clf.predict(pd.DataFrame(np.matmul(X[idx == False], cp[1])))

pd.crosstab(dd.Class[idx == False], pred, rownames=['Actual'], colnames=['Predicted'])
accuracy_score(pd.factorize(dd.Class[idx == False])[0], pred)


```
