import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from utils import newReadData

X, y = newReadData(unit=list(range(52)), addAbsNode=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=True)

forest = ensemble.RandomForestClassifier(n_estimators=50, max_features=None, min_samples_leaf=5,  criterion='gini', random_state=None, n_jobs=-1)
forest.fit(X_train, y_train)

class_num = [0 for _ in range(22)]
sum = 0

for i in y_test:
   class_num[int(i)] += 1
class_T = [0 for _ in range(22)]

for x, y in zip(forest.predict(X_test), y_test):
    if x == y:
        sum += 1
        class_T[int(x)] += 1

ac = sum / len(y_test)
print("all ac：", ac)

ac = [0 for _ in range(22)]
for i in range(22):
    ac[i] = class_T[i]/class_num[i]
    print(i, " ac", ac[i])

awac = np.sum(ac) / 22
print("weight ac：", awac)