import numpy as np 

def sigmoid(x):
        sig = np.vectorize(lambda y:  (1 - 1 / (1 + math.exp(y))) if y < 0 else  (1 / (1 + math.exp(-y))))
        return sig(x)

class mlp:
    def __init__(self,input_nodes=0,learning_rate=.1):
        self.number_of_nodes = []
        if init_nodes > 0:
            self.number_of_nodes.append(init_nodes)
        self.weights = []
        self.biases = []
        self.functions = []
        self.learning_rate = learning_rate

    def feed_forward(self,input_data):
        outputs = [np.matrix(input_data).T]

        for i in range(len(self.number_of_nodes) - 1):
            outputs.append(sigmoid((np.dot(self.weights[i], outputs[-1]) + self.biases[i])))

        return outputs