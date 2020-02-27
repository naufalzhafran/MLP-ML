from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
import random

iiris = load_iris()
clf = MLPClassifier(solver='sgd', batch_size=5, alpha=1e-5, hidden_layer_sizes=(6, 6), learning_rate_init=1, activation='logistic')

#clf.fit(iiris.data, iiris.target)
# clf.predict(iiris.data)
# print(dir(clf))

test_data = []
test_target = []
test_predict = []

training_data = []
training_target = []

idx = []

for i in range(30):
    k = random.randrange(0, 150, 1)
    idx.append(k)
    test_data.append(iiris.data[k])
    test_target.append(iiris.target[k])

for i in range(150):
    if i not in idx:
        training_data.append(iiris.data[i])
        training_target.append(iiris.target[i])

clf.fit(training_data, training_target)

test_predict = clf.predict(test_data)

print(test_target)
print(test_predict)