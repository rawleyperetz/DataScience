# K-nearest algorithm
import numpy as np 
from collections import Counter 

class knearest:
    #Initializing the training data
    def __init__(self,X,y,cols):
        self.cols = cols 
        self.X = np.array(X).reshape(-1,cols,order='F')
        #If training ground truth is string, use numbers to represent
        if isinstance(y[0],str) == True:
            uniq_elem = list(set(y))
            dic_elem = {uniq_elem[i]:i for i in range(len(uniq_elem))}
            self.y = [dic_elem[i] for i in y] 
            print(dic_elem)
        else: 
            self.y = y
    
    
    def fit(self, test_x, metric):
        '''
            This function takes test_x and metric value
            Metric value 1 == Manhattan, 2 == euclidean == l2 norm etc
        '''
        self.test_x = test_x
        #Creates a huge matrix with the column distance appended
        if metric: 
            num_rows = self.X.shape[0]
            sum = 0
            self.dist = []
            for i in range(num_rows):
                for j in range(self.cols):
                    sum = sum + pow((self.X[i][j]-test_x[j]),metric)
                self.dist.append(np.power(sum,(1/metric)))
                sum = 0
        self.data = np.c_[self.X, np.array(self.y).reshape(num_rows,-1) ,np.array(self.dist).reshape(num_rows,-1)]
        return self.data 
    
    #Classifies the new data point using the counter to find the most frequent element
    def predict(self,k):
        self.k = k 
        sort_data = sorted(self.dist)[:self.k]
        r_select = []
        for i in range(self.k):
            r_select.append(self.data[:,-2][self.data[:,-1] == sort_data[i]])
        r_select = np.array(r_select)
        class_r = Counter(list(r_select.flatten())).most_common(1)[0][0]
        return class_r
        

if __name__=='__main__':
    # sample = knearest(X=[7,7,3,1,7,4,4,4],y=['bad','bad','good','good'],cols=2)
    # sample.fit(test_y=[3,7],metric=2)
    # print(sample.predict(k=3))
    trial = knearest(X=[33.6,26.6,23.4,43.1,35.3,35.9,36.7,25.7,23.3,31,
                      50,30,40,67,23,67,45,46,29,56], y=[1,0,0,0,1,1,1,0,0,1],cols=2)
    print(trial.fit(test_x=[43.6,40], metric=2))
    print(trial.predict(k=5))





        





        
        
        