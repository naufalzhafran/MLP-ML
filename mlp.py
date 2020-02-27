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

training_input = [[0,0],[1,0],[0,1],[1,1]]
training_output = [[0,0],[1,1],[1,1],[0,0]]


def sigmoid(x):
        sig = np.vectorize(lambda y:  (1 - 1 / (1 + math.exp(y))) if y < 0 else  (1 / (1 + math.exp(-y))))
        return sig(x)

class mlp:
    def __init__(self,input_nodes=0,learning_rate=.1,n_batch=1):
        self.number_of_nodes = []
        if input_nodes > 0:
            self.number_of_nodes.append(input_nodes)
        self.weights = []
        self.biases = []
        self.functions = []
        self.learning_rate = learning_rate
        self.n_batch = n_batch

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

    def train(self, inp, targets,epoch=5,error_threshold=5):
        for n in range(0,epoch):
            sum_error = 0
            for x in range(len(inp)):
                target = np.matrix(targets[x]).T
                # Calculate output with given input
                outputs = self.feed_forward(inp[x])

                # Calculate each layer error
                errors = [np.subtract(target, outputs[-1])]
                sum_error += np.sum(np.power(np.subtract(target, outputs[-1]),2))
                for i in range(len(self.weights) - 1):
                    errors.insert(0, np.dot(self.weights[-1-i].T, errors[0]))

                for i in range(len(self.weights)):
                    # Calculate gradient and weight correction
                    gradient = np.multiply(errors[-1-i], np.multiply(outputs[-1-i] , (1-outputs[-1-i] )))
                    gradient *= self.learning_rate
                    self.biases[-1-i] += gradient
                    delta_w  = np.dot(gradient, outputs[-2-i].T)
                    self.weights[-1-i] += delta_w
            print("error :",sum_error/2,"Progress : ",n,"/",epoch)
        


if __name__ == "__main__":
    model = mlp(2)
    model.add_layer(2)
    model.add_layer(2)

    model.train(training_input,training_output)
    print(training_xor[0]['input'])
    output = model.feed_forward(training_xor[0]['input'])
    print(output[-1])
