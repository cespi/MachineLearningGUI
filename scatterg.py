def scatterg(x,y,label,num_labels,x1_axis,x2_axis,ptitle):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as palette
   
    label=np.asarray(label)
    label=label.astype(int)
    # Create palette    
    palette = palette.rainbow(np.linspace(0,1,num_labels + 1));
    colors = palette[label.astype(np.int64)+1,:];
    
    plt.figure()
    plt.scatter(x,y,c=colors)
    plt.title(ptitle)
    plt.xlabel(x1_axis)
    plt.ylabel(x2_axis)
