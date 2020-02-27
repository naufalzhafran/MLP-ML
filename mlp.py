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
            self.number_of_nodes.append(input_nodes) # add input layer
        self.weights = []
        self.delta_weights = []
        self.biases = []
        self.delta_biases = []
        self.learning_rate = learning_rate
        self.n_batch = n_batch # Belum dipakai
        self.number_of_layer = 0

    def add_layer(self, number_of_nodes: int,function="sigmoid"):
        '''
        Adding layer(Hidden or Output)
        '''
        self.number_of_nodes.append(number_of_nodes)

        if len(self.number_of_nodes) > 1:
            self.weights.append(np.random.randn(self.number_of_nodes[-1], self.number_of_nodes[-2]) )
            self.delta_weights.append(np.zeros((number_of_nodes,self.number_of_nodes[self.number_of_layer])).tolist())

        if len(self.number_of_nodes) > 1:
            self.delta_biases.append(np.zeros((number_of_nodes)).tolist())
            self.biases.append(np.random.uniform(0, 0, size=(number_of_nodes, 1)))

        self.number_of_layer += 1

    def feed_forward(self,input_data):
        '''
        Feed foward data, outputs = array of output for every layer, output[-1] = output of last layer/output layer
        '''
        outputs = [np.matrix(input_data).T]

        for i in range(len(self.number_of_nodes) - 1):
            outputs.append(sigmoid((np.dot(self.weights[i], outputs[-1]) + self.biases[i])))

        return outputs

    def train(self, inp, targets,epoch=5,error_threshold=5):
        '''
        Train the mlp
        '''
        counter = 0
        for n in range(0,epoch):
            sum_error = 0
            for x in range(len(inp)):
                counter += 1
                target = np.matrix(targets[x]).T
                # Calculate output with given input
                outputs = self.feed_forward(inp[x])

                # Calculate each layer error
                errors = [np.subtract(target, outputs[-1])]
                sum_error += np.sum(np.power(np.subtract(target, outputs[-1]),2)) # Calculate square error for every data
                for i in range(len(self.weights) - 1):
                    errors.insert(0, np.dot(self.weights[-1-i].T, errors[0]))

                """
                for i in range(len(self.weights)):
                    # Calculate gradient and weight correction
                    gradient = np.multiply(errors[-1-i], np.multiply(outputs[-1-i] , (1-outputs[-1-i] ))) # Calculate gradient for sigmoid activation function
                    gradient *= self.learning_rate
                    self.biases[-1-i] += gradient # Update biases
                    delta_w  = np.dot(gradient, outputs[-2-i].T)
                    self.weights[-1-i] += delta_w
                """
                
                self.update_delta_weights(outputs,target)

                if counter>=self.n_batch or x==len(inp)-1:
                    self.update_weights()
                    counter = 0

            print("error :",sum_error/2,"Progress : ",n,"/",epoch)
    
    def delta(self,layer,t_node,s_node,output,target):
        """
        Implementation of delta rule for weight a to b => w(s_node)(t_node)

        dEtotal/dw[s_node][t_node] = (dEtotal/dout[t_node]) x (dout[t_node]/dnet[t_node]) * (dnet[t_node]/dw[s_node][t_node])
        dEtotal/dw[s_node][t_node] = delta[t_node] * output[s_node]
                
        below is the implementation of delta[t_node] = (dEtotal/dout[t_node]) x (dout[t_node]/dnet[t_node])
        """
        #if at output layer, do delta rule 
        if layer==self.number_of_layer:
            return -(float(target[t_node])-output[layer].item(t_node)) * output[layer].item(t_node) * (1-output[layer].item(t_node))
        
        #else, do recursive until output layer 
        else:        
            total = 0
            for i in range(len(output[layer+1])):
                    total += self.delta(layer+1,i,t_node,output,target)*self.weights[layer].item((i,t_node))

            total = total*output[layer].item(t_node)*(1-output[layer].item(t_node))

            return total

    def update_delta_weights(self,output,target):
        """
        save each weight update value to delta_weight, when need to update to weight call func update_weight
        delta_weight will be set to 0
        """
        for i in range(0,self.number_of_layer):
            for j in range(len(self.delta_weights[i])):
                for k in range(len(self.delta_weights[i][j])):
                    self.delta_weights[i][j][k] += self.delta(i+1,j,k,output,target)*output[i].item(k)
                self.delta_biases[i][j] += self.delta(i+1,j,-1,output,target)
    
    def update_weights(self):
        """
        weight being updated by the value of delta_weight
        """
        for i in range(self.number_of_layer):
            for j in range(len(self.weights[i])):
                for k in range(len(self.delta_weights[i][j])):
                    self.weights[i][j][k] -= self.learning_rate*self.delta_weights[i][j][k]
                    self.delta_weights[i][j][k] = 0
                self.biases[i][j] -= self.learning_rate*self.delta_biases[i][j]
                self.delta_biases[i][j] = 0
        


if __name__ == "__main__":
    model = mlp(2,.25,2)
    model.add_layer(2)
    model.add_layer(2)

    model.train(training_input,training_output,50)
    print(training_xor[0]['input'])
    output = model.feed_forward(training_xor[0]['input'])
    print(output[-1])
