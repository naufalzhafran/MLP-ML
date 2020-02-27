from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from mlp import mlp

if __name__ == "__main__":
    model = mlp(4,0.075,5)
    model.add_layer(6)
    model.add_layer(6)
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
    for y in y_test:
        target = [0]*3
        target[y] = 1
        test_target.append(target)
    
    model.train(x_train,train_target,50)
    res = model.predict(x_test[0])

    print(res)