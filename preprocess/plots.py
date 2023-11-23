import matplotlib.pylab as plt
import numpy as np
import glob
file_list = glob.glob("data/open_specy/web/csv_created/Others/*.csv")
plt.rcParams['font.size'] = 18
# print(1%4)
for i in np.arange(0,len(file_list),9):
    fig,ax = plt.subplots(3,3, figsize=(15, 15))
    # ax.set_xlabel('Active Cdc2-cyclin B', fontsize = 20)
    rc = -1
    rcol = -1
    for file in file_list[i:i+9]:
        y = np.genfromtxt(file, delimiter=",")
        x = np.arange(4000)
        rcol+=1
        if rcol%3==0:
            rc+=1
        
        ax[rc,rcol%3].plot(y[:,0],y[:,1])
    # ax.tick_params(axis='both', which='major', labelsize=12)
    # ax.tick_params(axis='both', which='minor', labelsize=8)
    fig.savefig("preprocess_plots/open_spcey/raw/Others"+str(i)+".png")

