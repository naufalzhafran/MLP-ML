from sklearn.datasets import load_iris
from mlp import mlp

if __name__ == "__main__":
    model = mlp(4,0.01)
    model.add_layer(6)
    model.add_layer(6)
    model.add_layer(3)

    data = load_iris()    
    targets = []
    for x in data.target:
        target = [0]*3
        target[x] = 1
        targets.append(target)
    model.train(data.data,targets,1000)
    res = model.feed_forward(data.data[1])
    print(res[-1].T,targets[1])