import numpy as np

# Data
# X: [Hours Studied, Hours Slept]
# y: [Test Score]
X = np.array(([2, 9], [1, 5], [3, 6]), dtype = float)
y = np.array(([92], [86], [89]), dtype = float)

xPredicted = np.array(([0, 0]), dtype = float)

# Normalize our data
X = X / np.amax(X, axis = 0)
y = y / 100

xPredicted = xPredicted / np.amax(X, axis = 0)

# Activation Function
def sigmoid(t):
	return 1/(1 + np.exp(-t))

# Derivative of Activation Function
def sigmoid_derivative(p):
	return p * (1 - p)

class NeuralNetwork:
	def __init__(self, x, y):
		self.input = x
		self.weights1 = np.random.randn(self.input.shape[1], 3)
		self.weights2 = np.random.randn(3, 1)
		self.y = y
		self.output = np.zeros(y.shape)
		
	def feedforward(self):
		# Matrix Multiplication for each layer
		self.layer1 = sigmoid(np.dot(self.input, self.weights1))
		self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
		return self.layer2
	
	def backprop(self):
		d_weights2 = np.dot(self.layer1.T, 2*(self.y -self.output)*sigmoid_derivative(self.output))
		d_weights1 = np.dot(self.input.T, np.dot(2*(self.y -self.output)*sigmoid_derivative(self.output), self.weights2.T)*sigmoid_derivative(self.layer1))
		
		# Update Weights based on slope of loss function
		self.weights1 += d_weights1
		self.weights2 += d_weights2
		
	def train(self, X, y):
		self.output = self.feedforward()
		self.backprop()
		
	def predict(self):
		self.layer1 = sigmoid(np.dot(xPredicted, self.weights1))
		self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))

		print ("Predicted Data based on trained weights:")
		print ("Input: \n" + str(xPredicted))
		print ("Output: \n"+ str(self.layer2))
		print ("\n")

NN = NeuralNetwork(X, y)

for i in range(80000):
	if i % 20000 == 0:
		print ("# " + str(i) + "\n")
		print ("Input (Scaled): \n" + str(X))
		print ("Actual Output: \n" + str(y))
		print ("Predicted Output: \n" + str(NN.feedforward()))
		print ("Loss: \n" + str(np.mean(np.square(y - NN.feedforward()))))
		print ("\n")
		
	NN.train(X, y)

NN.predict()








