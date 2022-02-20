def neuronal_network(X,y,num_labels,hidden_size,learning_rate,max_iter):
    import numpy as np       
    from sklearn.preprocessing import OneHotEncoder  
    
    X = np.matrix(X)  
    y = np.matrix(y)
    encoder = OneHotEncoder(sparse=False)     
    y_2D=y.reshape(-1,1)
    y_onehot = encoder.fit_transform(y_2D)  

    #define the sigmoid function 
    def sigmoid(z):  
        return 1 / (1 + np.exp(-z))
    
    #forward propagation for two layers    
    def forward_propagate(X, theta1, theta2):  
        m = X.shape[0]
    
        a1 = np.insert(X, 0, values=np.ones(m), axis=1)
        z2 = a1 * theta1.T
        a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
        z3 = a2 * theta2.T
        h = sigmoid(z3)
    
        return a1, z2, a2, z3, h
    
    def sigmoid_gradient(z):  
        return np.multiply(sigmoid(z), (1 - sigmoid(z)))
    
    def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):  
        ##### this section is identical to the cost function logic we already saw #####
        m = X.shape[0]
        X = np.matrix(X)
        y = np.matrix(y)
    
        # reshape the parameter array into parameter matrices for each layer
        theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
        theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
        #print('X ',X.shape,' t1 ',theta1.shape,' t2 ',theta2.shape)
        # run the feed-forward pass
        a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    
        # initializations
        J = 0
        delta1 = np.zeros(theta1.shape)  # (25, 401)
        delta2 = np.zeros(theta2.shape)  # (10, 26)
    
        # compute the cost
        for i in range(m):
            first_term = np.multiply(-y[i,:], np.log(h[i,:]))
            second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
            J += np.sum(first_term - second_term)
    
        J = J / m
    
        # add the cost regularization term
        J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
    
        ##### end of cost function logic, below is the new part #####
    
        # perform backpropagation
        for t in range(m):
            a1t = a1[t,:]  
            z2t = z2[t,:] 
            a2t = a2[t,:]  
            ht = h[t,:]  
            yt = y[t,:]  
    
            d3t = ht - yt  
    
            z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
            d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)
    
            delta1 = delta1 + (d2t[:,1:]).T * a1t
            delta2 = delta2 + d3t.T * a2t
    
        delta1 = delta1 / m
        delta2 = delta2 / m
    
        # add the gradient regularization term
        delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * learning_rate) / m
        delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * learning_rate) / m
    
        # unravel the gradient matrices into a single array
        grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    
        return J, grad

    # initial setup
    input_size=X.shape[1]  #Assign the number of columns  
    
    # randomly initialize a parameter array of the size of the full network's parameters
    params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25 
               
    J, grad = backprop(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)  
    
    # minimize the objective function
    from scipy.optimize import minimize
    fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),  
                method='TNC', jac=True, options={'maxiter': max_iter})
    return fmin
    
def neuronal_network_predict(X,fmin,hidden_size,num_labels):
    import numpy as np
    
    #define the sigmoid function 
    def sigmoid(z):  
        return 1 / (1 + np.exp(-z))    
    #forward propagation for two layers    
    def forward_propagate(X, theta1, theta2):  
        m = X.shape[0]    
        a1 = np.insert(X, 0, values=np.ones(m), axis=1)
        z2 = a1 * theta1.T
        a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
        z3 = a2 * theta2.T
        h = sigmoid(z3)    
        return a1, z2, a2, z3, h
        
    input_size=X.shape[1]  #Assign the number of columns     

    theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))  
    theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)  
    
    pred_label = np.array(np.argmax(h, axis=1)) 
    pred_label_1d=pred_label[:,0]
    return pred_label_1d

def score(label,pred_label):    
    correct = [1 if a == b else 0 for (a, b) in zip(pred_label, label)]  
    accuracy = (sum(map(int, correct)) / float(len(correct)))  
    print ('accuracy = {0}%'.format(accuracy * 100))    
    return accuracy    
     

