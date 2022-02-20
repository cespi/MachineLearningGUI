def scatterg(x1,x2,x3,label,num_labels,x1_axis,x2_axis,x3_axis,ptitle):
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import matplotlib.cm as palette

    label=np.asarray(label)
    label=label.astype(int)

    # Create palette
    palette = palette.rainbow(np.linspace(0,1,num_labels + 1));
    colors = palette[label.astype(np.int64)+1,:];

    fig=plt.figure()    
    ax = fig.add_subplot(111, projection='3d')    
    ax.scatter(x1,x2,x3,label,c=colors)
    plt.title(ptitle)
    plt.xlabel(x1_axis)
    plt.ylabel(x2_axis)
    ax.set_zlabel(x3_axis)