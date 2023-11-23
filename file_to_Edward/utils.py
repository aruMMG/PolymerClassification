import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def createFolders(args):
    """Creates required forlders for GAN training"""
    log_name = args.log_name
    if os.path.exists(os.path.join("logFile",args.log_name)):
        exp_num = 1
        
        while exp_num<1000:
            log_name = args.log_name + f"_{exp_num}"
            if os.path.exists(os.path.join("logFile",log_name)):
                exp_num+=1
                continue
            else:
                os.mkdir(os.path.join("logFile",log_name))
                break
    else:
        os.mkdir(os.path.join("logFile",log_name))
    os.mkdir(os.path.join("logFile",log_name, 'weights'))
    
    return log_name


def plot9Data(x, savename, label=None, all=False, name=None):
    y = np.arange(4000)
    c=0
    if all:
        for i in np.arange(0,len(x),16):
            r=-1
            c+=1
            fig, ax = plt.subplots(4,4,figsize=(15,15))
            for j in range(16):
                if j%4==0:
                    r+=1
                if (i+j)<len(x):
                    ax[r,j%4].plot(y,x[i+j,:])
                else:
                    break
                if isinstance(name, np.ndarray):
                    ind = i+j
                    ax[r,j%4].text(20,0.9, str(name[ind]))
            fig.savefig(savename+str(c)+".jpg")
    else:
        fig, ax = plt.subplots(3,3,figsize=(15,15))
        r = -1
        for i in range(9):
            if i%3==0:
                r+=1
            if i>=len(x):
                break
            ax[r,i%3].plot(y,x[i,:])
            if isinstance(label, np.ndarray):
                ax[r,i%3].text(20,0.5, str(np.argmax(label[i])))
            if isinstance(name, np.ndarray):
                ax[r,i%3].text(20,0.9, str(name[i]))
                
        fig.savefig(savename+str(c)+".jpg")

def plotDataXY(x, y, savename, label=None, all=False, name=None):
    c=0
    if all:
        for i in np.arange(0,len(x),16):
            r=-1
            c+=1
            fig, ax = plt.subplots(4,4,figsize=(15,15))
            for j in range(16):
                if j%4==0:
                    r+=1
                if (i+j)<len(x):
                    ax[r,j%4].plot(y[i+j],x[i+j])
                else:
                    break
                if isinstance(name, np.ndarray):
                    ind = i+j
                    ax[r,j%4].text(1000,0.03, str(name[ind]))
            fig.savefig(savename+str(c)+".jpg")
    else:
        fig, ax = plt.subplots(3,3,figsize=(15,15))
        r = -1
        for i in range(9):
            if i%3==0:
                r+=1
            if i>=len(x):
                break
            ax[r,i%3].plot(y[i],x[i])
            if isinstance(label, np.ndarray):
                ax[r,i%3].text(0.5,0.5, str(np.argmax(label[i])))
            if isinstance(name, np.ndarray):
                ax[r,i%3].text(0.5,0.9, str(name[i]))
                
        fig.savefig(savename+str(c)+".jpg")

def asls(x, y, lambda1, lambda2):
    # Initialize weights
    w = np.ones(len(x))

    # Define objective function
    def objective(params):
        b = params[0]
        z = params[1:]
        residuals = y - b - z
        # Assign asymmetric weights
        w[residuals > 0] = lambda1
        w[residuals < 0] = lambda2
        return np.sqrt(w * residuals**2)

    # Perform least squares optimization
    result = least_squares(objective, [np.mean(y)] + [0]*len(x), bounds=([-np.inf] + [-np.inf]*len(x), [np.inf] + [np.inf]*len(x)),
                           method='trf', loss='linear', args=())

    # Return baseline-corrected signal
    return y - result.x[0] - result.x[1:]

if __name__=="__main__":
    # Plotting ==========================
    # x = np.load("data/real/other_norm_no_neg_data.npy")
    # name_all = np.load("data/real/FTIR_name.npy")
    # name_notPolymer = np.array([i for i in name_all if i.startswith("notPolymer")])
    # plot9Data(x, "plots/real/check", all=True, name=name_notPolymer)

    # Baseline correction =================================
    # x, y = np.loadtxt('ir_data.txt', delimiter=',')
#     arr = np.genfromtxt("data/Data_Edward/Data/FTIR/HDPE/HDPE02.CSV", delimiter=",")
# # Perform ASLS baseline correction
#     import pybaselines
#     base = pybaselines.whittaker.airpls(arr[:,1], lam=10)
#     # baseline_corrected = asls(arr[:,0], arr[:,1], 100, 1)

#     # Plot original and baseline-corrected data
#     plt.plot(arr[:,0], arr[:,1], label='Original data')
#     plt.plot(arr[:,0], base[0], label='Baseline-corrected data')
#     # plt.plot(arr[:,0], baseline_corrected, label='Baseline-corrected data')
#     plt.legend()
    plt.savefig("check.png")