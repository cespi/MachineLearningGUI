def Xgather(X):
    import numpy as np
    import flager as sf
    np.asarray(X)
    gathers=X[:,0]
    gatherref=0
    u_gather=np.array([])
    for gather in gathers:
    	if gather != gatherref:
    	    u_gather=np.append(u_gather,gather)
    	    gatherref=gather

    Xgather=np.zeros([len(u_gather),9])
    for a in np.arange(len(u_gather)):
    	idx=( X[:,0]==u_gather[a] )
    	Xgather[a,0]=u_gather[a]	      # gather
    	Xgather[a,1]=np.mean(X[idx,1])    # mean feature 1
    	Xgather[a,2]=np.mean(X[idx,2])    # mean featrue2
    	Xgather[a,3]=np.mean(X[idx,3])    # mean feature3
    	Xgather[a,4]=np.mean(X[idx,4])    # mean feature4
    	Xgather[a,5]=np.min(X[idx,5])     # min feature5
    	Xgather[a,6]=sf.flagger(X[idx,4]) # flagger feature 4
    	Xgather[a,7]=np.mean(X[idx,7])    # mean feature 7
    	Xgather[a,8]=np.mean(X[idx,8])    # mean feature 8
    return Xgather
