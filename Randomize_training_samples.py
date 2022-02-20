def randomTrain(X,label,train_size_pct):
    import numpy as np

    train_size=int(len(X)/100*train_size_pct)
    Xtrain=np.zeros((train_size,2))           #Initialize train matrix
    randidx = np.random.permutation(len(X)) #Randomly reorder the indices of examples

    Xtrain = X[randidx[0:train_size],:]     #Take the first K samples as training examples
    labeltrain = np.zeros((train_size+1,1))           #Initialize train matrix
    labeltrain = label[randidx[0:train_size]]       #Get the corresponding user labels for QC

    X_xval = X[randidx[train_size:len(X)],:]     #Take the first K samples as training examples
    label_xval = np.zeros((len(X)-train_size+1,1))           #Initialize train labels
    label_xval = label[randidx[train_size:len(X)]]       #Get the corresponding user labels for QC

    return Xtrain,labeltrain,X_xval,label_xval
