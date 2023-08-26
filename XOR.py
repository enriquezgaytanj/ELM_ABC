import numpy as np


class ELM (object):
    def __init__(self, inputSize,outputSize, hiddenSize):
        """
        Inicialización de weight, bias, input layer, and hidden layers

        """
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize   

        # Initialize random weight with range [-0.5, 0.5]
        self.weight = np.matrix(np.random.uniform(-0.5, 0.5, (self.hiddenSize, self.inputSize)))

        # Initialize random bias with range [0, 1]
        self.bias = np.matrix(np.random.uniform(0, 1, (1, self.hiddenSize)))
        
        self.H = 0
        self.beta = 0

    def sigmoid(self,x):
        return 1 / (1+ np.exp(-1*x))
    
    
    def predict(self, X):
        X = np.matrix(X)
        y = self.sigmoid((X * self.weight.T) + self.bias) * self.beta

        return y

    def train(self, X, y):
        """
        Extreme Learning Machine training process
        Parameters:
        X: array-like or matrix
            Training data that contains the value of each feature
        y: array-like or matrix
            Training data that contains the value of the target (class)
        Returns:
            The results of the training process   
        """

        X = np.matrix(X)
        y = np.matrix(y)        
        
        # Calculate hidden layer output matrix (Hinit)
        self.H = (X * self.weight.T) + self.bias

        # Sigmoid activation function
        self.H = self.sigmoid(self.H)

        # Calculate the Moore-Penrose pseudoinverse matriks        
        H_moore_penrose = np.linalg.inv(self.H.T * self.H) * self.H.T

        # Calculate the output weight matrix beta
        self.beta = H_moore_penrose * y

        return self.H * self.beta
    



'''
Pruebas
'''

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# Create random classification datasets with 1000 samples
#data = datasets.make_classification(10, n_features=4)
data = (([[0,0], [0,1], [1,0], [1,1]]),([1,0,0,1]))

print(data)
print("--------------------------------------------------------------")

#input_No = data[0].shape[1] # 20 datos de prueba 
#print(input_No)
# Create instance of ELM object with 10 hidden neuron

elm = ELM ( 2, 1, 10)


# Train test split 80:20
X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.2)
y_train=np.array(y_train)
print(y_train)
print("--------------------------------------------------------------")
print(X_train)
print("--------------------------------------------------------------")
# Train data

elm.train(X_train,y_train.reshape(-1,1))



# Make prediction from training process
y_pred = elm.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

y_pred1 = np.zeros((len(y_pred),), dtype=int)


i=0
for ii in y_pred:
    y_pred1[int(i)]=int(ii)
    i=i+1    
    


print('Accuracy: ', accuracy_score(y_test, y_pred1))

print(y_test)
print(y_pred1)
'''





print('Accuracy: ', accuracy_score(y_test, y_pred))




print(data)
print("--------------------------------------------------------------")
print("Dato de interes")
print(data[0].shape[1])
print("--------------------------------------------------------------")
#onces elm = ELM(data[0].shape[1], 1, 10)


plt.plot( data, linestyle='--')
plt.title("Matplotlib Plot NumPy Array As Line")
plt.xlabel("x axis") 
plt.ylabel("y axis") 
plt.show()
'''
