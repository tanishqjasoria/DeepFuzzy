class deepfuzzy:
    W = dict()
    b = dict()
    caches = []
    grads = dict()
    def __init__(self,layers, test_data, train_data):
        self.layers = layers
        self.test_data = test_data
        self.train_data = train_data
        self.L = len(layers)
        
    def initialize(self, initializer = 'random'):
        for i in range(1,len(self.layers)):
            self.W[str(i)] = np.random.randn(self.layers[i],self.layers[i-1])
            self.b[str(i)] = np.random.randn(self.layers[i],1)

    @staticmethod
    def forward(A,W,b):
        Z = np.dot(W,A) + b
        cache = (A, W, b)
        return Z, cache
    @staticmethod        
    def forwardAct(A,W,b,activation = 'sigmoid'):
        Z, liner_cache = deepfuzzy.forward(A,W,b)
        activation_cache = Z
        if activation == 'sigmoid':
            A_next = sigmoid(Z)
        if activation == 'relu':
            A_next = relu(Z)
        return A_next, (linear_cache, activation_cache)
    
    def forwardProp(self):
        A = self.train_data
        for i in range (1,self.L):
            A, self.cache = deepfuzzy.forwardAct(A, self.W[str(i)], self.b[str(i)], activation = 'relu')
    
    def saveWeights(self):
        np.save('W.npy',  self.W)  
        np.save('b.npy', self.b)
        my_dict_back = np.load('my_dict.npy')
        
    def loadWeights():
        self.W = np.load('W.npy')
        self.b = np.load('b.npy')

    @staticmethod    
    def backward(dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = 1.0/m * dZ.dot(A_prev.T)
        db = 1.0/m * np.sum(dZ,axis=1,keepdims=True)
        dA_prev = W.T.dot(dZ)
        return dA_prev, dW, db
    @staticmethod
    def backwardAct(dA, cache, activation):
        linear_cache, activation_cache = cache
        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)
            dA_prev, dW, db = deepfuzzy.backward(dZ, linear_cache)
        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = deepfuzzy.backward(dZ, linear_cache)
        return dA_prev, dW, db
    
    def backprop(self,AL, Y, caches):
        L = self.L
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) 
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        current_cache = self.caches[L-1]
        self.grads["dA" + str(L)], self.grads["dW" + str(L)], self.grads["db" + str(L)] = deepfuzzy.backwardAct(dAL, current_cache, "sigmoid")
        for l in reversed(range(L-1)):
            current_cache = self.caches[l]
            dA_prev_temp, dW_temp, db_temp = deepfuzzy.backwardAct(self.grads["dA"+str(l+2)], current_cache, "relu")
            self.grads["dA" + str(l + 1)] = dA_prev_temp
            self.grads["dW" + str(l + 1)] = dW_temp
            self.grads["db" + str(l + 1)] = db_temp

    @staticmethod
    def compute_cost(AL, Y):
        m = Y.shape[1]
        cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
        cost = np.squeeze(cost) 
        return cost

    def update_parameters(self,learning_rate):
        for l in range(self.L):
            self.W[str(l+1)] = self.W[str(l+1)] - learning_rate * self.grads["dW" + str(l+1)]
            self.b[str(l+1)] = self.b[str(l+1)] - learning_rate * self.grads["db" + str(l+1)]
            

    def predict(self):
            m = X.shape[1]
            n = self.L
            p = np.zeros((1,m))
            probas, caches = L_model_forward()
            for i in range(0, probas.shape[1]):
                if probas[0,i] > 0.5:
                    p[0,i] = 1
                else:
                    p[0,i] = 0
            
            #print results
            #print ("predictions: " + str(p))
            #print ("true labels: " + str(y))
            print("Accuracy: "  + str(np.sum((p == y)/m)))
                
            return p