from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

dd = pd.read_csv("/Users/eric.golinko/Dropbox/Research/ISF 2018/GFEL_data/cnae.csv")

X = pd.get_dummies(dd.drop("Class", axis = 1))
# X = pd.get_dummies(dd.drop("Class", axis = 1).astype('object'))
X["Class"] = dd.Class

clf = GaussianNB()
clf.fit(X = pd.DataFrame(cp[0][idx == True]), y = pd.factorize(cp[2].Class[idx == True])[0])
pred = clf.predict(pd.DataFrame(np.matmul(cp[2].drop("Class", axis = 1)[idx == False], cp[1])))

pd.crosstab(cp[2].Class[idx == False], pred, rownames=['Actual'], colnames=['Predicted'])
accuracy_score(pd.factorize(cp[2].Class[idx == False])[0], pred)
