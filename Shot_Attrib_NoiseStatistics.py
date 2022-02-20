def Xshot(X):
    import numpy as np
    import flager as sf
    np.asarray(X)
    shots=X[:,0]
    shotref=0
    u_shot=np.array([])
    for shot in shots:
    	if shot != shotref:
    	    u_shot=np.append(u_shot,shot)
    	    shotref=shot

    Xshot=np.zeros([len(u_shot),9])
    for a in np.arange(len(u_shot)):
    	idx=( X[:,0]==u_shot[a] )
    	Xshot[a,0]=u_shot[a]	       #shot
    	Xshot[a,1]=np.mean(X[idx,1])   #mean trace
    	Xshot[a,2]=np.mean(X[idx,2])   #mean offset
    	Xshot[a,3]=np.mean(X[idx,3])   #mean boat speed
    	Xshot[a,4]=np.mean(X[idx,4])   #mean RMS
    	Xshot[a,5]=np.min(X[idx,5])   #min dom freq
    	Xshot[a,6]=sf.flagger(X[idx,4])   #mean bad_flag
    	Xshot[a,7]=np.mean(X[idx,7])   #max max Offset
    	Xshot[a,8]=np.mean(X[idx,8])   #mean User Flagmean RMS
    return Xshot
