from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
import random
from mlp import mlp
import numpy as np

if __name__ == "__main__":
    model = mlp(4,0.05,5)
    model.add_layer(6)
    model.add_layer(5)
    model.add_layer(3)

    data = load_iris()    
    targets = []
    x_train, x_test, y_train, y_test = train_test_split(data.data,data.target,test_size=0.2,random_state=100)
    train_target = []
    test_target = []
    for x in data.target:
        target = [0]*3
        target[x] = 1
        targets.append(target)
    for y in y_train:
        target = [0]*3
        target[y] = 1
        train_target.append(target)
    
    model.train(x_train,train_target,200)
    count = 0
    for x in range(len(x_test)):
        res = model.predict(x_test[x])
        if res[0] == y_test[x]:
            count +=1
    
    clf = MLPClassifier(solver='sgd', batch_size=5, hidden_layer_sizes=(6,5), learning_rate_init=.05, activation='logistic',max_iter=200)

    clf.fit(x_train, y_train)

    print("sklearn : ",clf.score(x_test, y_test))

    print("myMLP : ",count/len(x_test))