import numpy as np 
import math

training_xor = [
    {
        "input": [0, 0],
        "output": [0,0]
    },
    {
        "input": [1, 0],
        "output": [1,1]
    },
    {
        "input": [0, 1],
        "output": [1,1]
    },
    {
        "input": [1, 1],
        "output": [0,0]
    }
]


def sigmoid(x):
        sig = np.vectorize(lambda y:  (1 - 1 / (1 + math.exp(y))) if y < 0 else  (1 / (1 + math.exp(-y))))
        return sig(x)

class mlp:
    def __init__(self,input_nodes=0,learning_rate=.1):
        self.number_of_nodes = []
        if input_nodes > 0:
            self.number_of_nodes.append(input_nodes)
        self.weights = []
        self.biases = []
        self.functions = []
        self.learning_rate = learning_rate

    def add_layer(self, number_of_nodes: int,function="sigmoid"):
        self.number_of_nodes.append(number_of_nodes)

        if len(self.number_of_nodes) > 1:
            self.weights.append(np.random.randn(self.number_of_nodes[-1], self.number_of_nodes[-2]) * np.sqrt(2 / (self.number_of_nodes[-1] + self.number_of_nodes[-2])))
            self.functions.append(function)

        if len(self.number_of_nodes) > 1:
            self.biases.append(np.random.uniform(0, 0, size=(number_of_nodes, 1)))

    def feed_forward(self,input_data):
        outputs = [np.matrix(input_data).T]

        for i in range(len(self.number_of_nodes) - 1):
            outputs.append(sigmoid((np.dot(self.weights[i], outputs[-1]) + self.biases[i])))

        return outputs

if __name__ == "__main__":
    model = mlp(2)
    model.add_layer(2)
    model.add_layer(2)
    print(training_xor[0]['input'])
    output = model.feed_forward(training_xor[0]['input'])
    print(output[-1])
