import numpy as np
import matplotlib.pyplot as plt
import math as m
import random
import copy
# from sklearn.cluster import KMeans



def euclidianDistance(x, y):
    return m.sqrt(m.pow(abs(x[0] - y[0]),2) + m.pow(abs(x[1] - y[1]),2))

def sse(model, data):
    pred = model.predict(data)
    sse = 0
    for i in range(0, model.k):
        sse += np.sum([m.pow(euclidianDistance(model.centroids[i],data[x]),2) for x in [k for k in range(0,len(data)) if pred[k]==i]])
    return sse

class k_medoids:
    def __init__(self, k=2, thresold = 0.001, max_iter = 300, has_converged=False):
    
        ''' 
        Class constructor
        
        Parameters
        ----------
        - k: number of clusters. 
        - thresold (percentage): stop algorithm when difference between prev cluster 
                                and new cluster is less than thresold
        - max_iter: number of times centroids will move
        - has_converged: to check if the algorithm stop or not
        '''


        self.k = k
        self.thresold = thresold
        self.max_iter = max_iter
        self.has_converged= has_converged
        
    def initCentroids(self, X):
        ''' 
        Parameters
        ----------
        X: input data. 
        '''
        self.centroids=[]
        
        #Starting clusters will be random members from X set
        indexes = np.random.randint(0, len(X)-1,3)
        self.centroids=X[indexes]
            
    def fit(self, X):
        '''
        FIT function, used to find clusters
    
        Parameters
        ----------
        X: input data. 
        '''
        #Init list cluster centroids
        self.initCentroids(X)
        pickupPoint = copy.deepcopy(self.centroids)
        # for c in self.centroids:
        #     pickupPoint.append(c)
        # print(pickupPoint)
        while len(pickupPoint) < len(X):
            # print("{} & {}".format(len(pickupPoint),len(X)))
            cur_centroids = copy.deepcopy(self.centroids)
            prevSSE = sse(self,X)
            ranInt = random.randint(0,len(X)-1)
            ranPoint = X[ranInt]
            # print(ranPoint)
            # print(pickupPoint)
            while ranPoint in pickupPoint:
                ranInt = random.randint(0,len(X)-1)
                ranPoint = X[ranInt]
            # pickupPoint.append(ranPoint)
            pickupPoint = np.append(pickupPoint,ranPoint)
            ic = self.indexCentroidOfPoint(ranPoint)
            self.centroids[ic] = ranPoint
            currSSE = sse(self,X)
            if currSSE >= prevSSE:
                self.centroids[ic] = cur_centroids[ic]
        #Each cluster represented by its centroid
        return np.array(self.centroids)

    def indexCentroidOfPoint(self, point):
        return np.argmin([euclidianDistance(c,point) for c in self.centroids])
            

    def predict(self, data):
        ''' 
        Parameters
        ----------
        data: input data.
        
        Returns:
        ----------
        pred: list cluster indexes of input data 
        '''
    
        pred = []
        for i in range(len(data)):
            # Create list distances between centroids and data sample
            d_list = []
            for j in range(len(self.centroids)):
                # Calculate distances between current data sample and centroid(using euclidian distance) 
                # Store to d_list
                #TODO 
                d_list.append(euclidianDistance(self.centroids[j], data[i]))

            # Store the Cluster has minimal distance between its centroid and current data sample to pred
            #TODO 
            pred.append(d_list.index(min(d_list)))
            
        return np.array(pred)

def visualize(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    
    #you can fix this dpi 
    plt.figure(dpi=120)
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.plot()
    plt.show()

def main():
    np.random.seed(8) # This will fix the randomization; so, you and me will have the same results
    means = [[2, 2], [8, 3], [3, 6]]
    cov = [[1, 0], [0, 1]]
    N = 500
    X0 = np.random.multivariate_normal(means[0], cov, N)
    X1 = np.random.multivariate_normal(means[1], cov, N)
    X2 = np.random.multivariate_normal(means[2], cov, N)

    X = np.concatenate((X0, X1, X2), axis = 0)
    K = 3

    original_label = np.asarray([0]*N + [1]*N + [2]*N).T

    # test result KMeans lib
    # model2 = KMeans(n_clusters=3, random_state=0).fit(X)
    # print('Centers found by scikit-learn:')
    # print(model2.cluster_centers_)
    # pred_label = model2.predict(X)
    # visualize(X, pred_label)

    # test result k-means class
    model1=k_medoids(k=3)
    print('Centers found by your model:')
    print(model1.fit(X))
    pred=model1.predict(X)
    # print(sse(model1,X))
    visualize(X,pred)
main()
 
