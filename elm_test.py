import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import elm as ELM

# Create random classification datasets with 1000 samples
data = datasets.make_classification(100)

input_No = data[0].shape[1] # 20 datos de prueba 

# Create instance of ELM object with 10 hidden neuron
ElM_1=ELM(2,2,30)
elm2 = ELM ( input_No, 1, 10)

'''
# Train test split 80:20
X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.2)
'''

print(data)
print("--------------------------------------------------------------")
print("Dato de interes")
print(data[0].shape[1])
print("--------------------------------------------------------------")
#onces elm = ELM(data[0].shape[1], 1, 10)

'''
plt.plot( data, linestyle='--')
plt.title("Matplotlib Plot NumPy Array As Line")
plt.xlabel("x axis") 
plt.ylabel("y axis") 
plt.show()
'''
