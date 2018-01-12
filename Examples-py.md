Supervised
```

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

dd = pd.read_csv("https://s3-us-west-2.amazonaws.com/researchs/learn_w_cat_data/car.csv")

X = pd.get_dummies(dd.drop("Class", axis = 1))
X["Class"] = dd.Class

cp = cpir_gel(source_data_ = X, k = 10, learning_method = 'supervised', class_var = "Class")

idx = np.random.uniform(0, 1, len(cp[0])) <= .8

clf = GaussianNB()
clf.fit(X = pd.DataFrame(cp[0][idx == True]), y = pd.factorize(cp[2].Class[idx == True])[0])
pred = clf.predict(pd.DataFrame(np.matmul(cp[2].drop("Class", axis = 1)[idx == False], cp[1])))

pd.crosstab(cp[2].Class[idx == False], pred, rownames=['Actual'], colnames=['Predicted'])
accuracy_score(pd.factorize(cp[2].Class[idx == False])[0], pred)

```

Unsupervised

```
X = pd.get_dummies(dd.drop("Class", axis = 1))

cp_u = cpir_gel(source_data_ = X, k = 10, learning_method = 'unsupervised')

clf = RandomForestClassifier()
clf.fit(X = pd.DataFrame(cp_u[0][idx == True]), y = pd.factorize(dd.Class[idx == True])[0])
pred = clf.predict(pd.DataFrame(np.matmul(cp_u[2][idx == False], cp_u[1])))

pd.crosstab(dd.Class[idx == False], pred, rownames=['Actual'], colnames=['Predicted'])
accuracy_score(pd.factorize(cp[2].Class[idx == False])[0], pred)


```
