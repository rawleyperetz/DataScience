# Simple Perceptron Class For One Dimension

import numpy as np 

class SimplePerceptron:
    def __init__(self, X, y, weights, bias, num_cols=1) -> None:
        self.weights = np.array(weights)
        self.X = np.array(X).reshape(num_cols,-1)
        self.y = np.array(y) 
        self.bias = bias 
    
    # Sigmoid Activation Function 
    def sigmoid(self,X: np.ndarray):
        return 1/(1 + np.exp(-1*X))
    
    # Binary: 1 if > threshold value and 0 otherwise
    def threshold_bound(self, predicted: np.ndarray):
        predict = [1 if i >= self.threshold else 0 for i in list(predicted)] 
        return np.array(predict)
    
    def fit(self, threshold, learning_rate, iteration):
        self.threshold = threshold
        self.iter = iteration 
        self.lr = learning_rate
        self.pred_y = self.threshold_bound(self.sigmoid(self.weights@self.X + self.bias))
        i = 1
        while i <= self.iter:
            # Mean Absolute Error
            self.weights_loss = np.abs(self.y - self.pred_y)
            # Stochastic Gradient Descent for Weights
            self.weights = self.weights - self.lr*(self.pred_y - self.y)@self.X.transpose().sum(axis=1)
            # Stochastic Gradient Descent for Bias
            self.bias = self.bias - self.lr*(self.pred_y - self.y)
            self.pred_y = self.threshold_bound(self.sigmoid(self.weights@self.X + self.bias))
            print(f"Loss = {self.weights_loss}")
            print(f"New Weight = {self.weights}, New Bias = {self.bias}")
            i = i + 1 
        print('===============================================')
        print(f"{self.pred_y}")
        return f"Final weights = {self.weights}, Final Bias = {self.bias}"
    
    
    def accuracy(self):
        # Accuracy = sum of true positive and true negative over size of prediction
        self.tp_tn = np.count_nonzero(self.threshold_bound(self.weights_loss) == 0)
        acc = (self.tp_tn/self.pred_y.size)*100
        return acc
    

if __name__=='__main__':
    sim = SimplePerceptron(X=[0,0,1,1,0,1,0,1],y=[0,1,1,1],num_cols=2, weights=[0.9,0.9], bias=0)
    sim.fit(threshold=0.5, learning_rate=0.5, iteration=100)
    print(f"Accuracy is {sim.accuracy()}%")

        
        