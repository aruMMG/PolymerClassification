import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import pybaselines
import csv
import glob
import pandas as pd
import argparse

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

    # os.mkdir(os.path.join("logFile",log_name, 'weights_pre'))
    # os.mkdir(os.path.join("logFile",log_name, 'weights_trans'))
    # os.mkdir(os.path.join("logFile",log_name, 'weights_incep'))
    # os.mkdir(os.path.join("logFile",log_name, 'weights_incep_pre'))
    return log_name

def save_arguments_to_file(args, filename):
    with open(filename, 'w') as file:
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")

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
                    ax[r,j%4].text(0.1,np.max(x[i+j,:]), str(name[ind]))
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

def asls_eddie(old_array):
    return old_array - pybaselines.whittaker.asls(old_array)[0]
def asls_dataset(array, save_path=None):
    out_array = None
    for i in range(len(array)):
        new_array = asls_eddie(array[i])
        if out_array is None:
            out_array = new_array
        else:
            out_array = np.vstack((out_array, new_array))
    if len(out_array)!=len(array):
        print("Baseline correction for all array not available")
    if save_path:
        np.save(save_path, out_array)
    else:
        return out_array
def dpt2csv(dpt_path):
    """
    Inputs: 
        dpt_path: file path containing dpt files
    
    Save csv files to dpt_path with same name
    """
    for dpt_file in glob.glob(os.path.join(dpt_path, "*.dpt")):
    # Extract the filename without the extension
        filename = os.path.splitext(dpt_file)[0]

        # Create the output CSV file
        csv_file = f'{filename}.csv'

        # Read the .dpt file and extract the data
        data = []
        with open(dpt_file, 'r') as file:
            for line in file:
                # Split the line by comma to separate wavelength and spectroscopy values
                values = line.strip().split(',')
                data.append(values)
        reversed_data = data[::-1]

        # Write the extracted data to the CSV file
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            # writer.writerow(['Wavelength', 'Spectroscopy'])  # Write the header row
            writer.writerows(reversed_data)  # Write the data rows

def zero672(csv_path, save_dir):
    for csv_file in glob.glob(csv_path + "*.csv"):
        csv_name = os.path.basename(csv_file)
        data = pd.read_csv(csv_file, names=["c1", "c2"])
        data.loc[data['c1'] < 672, 'c2'] = 0
        data.to_csv(os.path.join(save_dir, csv_name), header=False, index=False)

def to4000(array, wavelength):
    out_array = None
    for i in range(len(array)):
        new_array = to4000_one(array[i], wavelength)
        if out_array is None:
            out_array = new_array
        else:
            out_array = np.vstack((out_array, new_array))
    if len(out_array)!=len(array):
        print("Baseline correction for all array not available")
    return out_array
def to4000_one(arr, wavelength):    
    interpolated_intensity = np.zeros((1,4000))
    for i in np.arange(400,4000):
        interpolated_intensity[0,i] = np.interp(i, wavelength, arr)
    return interpolated_intensity
def createNPY_raw(csv_path, save_name="csv.npy"):
    array = None
    for filename in os.listdir(csv_path):
        if filename.endswith(".csv"):
            dpt_path = os.path.join(csv_path, filename)
            
            # Read .dpt file and extract wavelength and intensity data
            data = np.genfromtxt(dpt_path, dtype=float, delimiter=",")
            data = data[data[:,0].argsort()]
            wavelength = data[:, 0]
            intensity = data[:, 1]
            if intensity.shape[0]==1867:
                if array is None:
                    array = intensity
                else:
                    array = np.vstack((array, intensity))
    save_path = os.path.join(csv_path, save_name)
    print(array.shape)
    np.save(save_path, array)

def createNPY(csv_path, save_name="csv.npy"):
    array = None
    for filename in os.listdir(csv_path):
        if filename.endswith(".CSV"):
            print(filename)
            dpt_path = os.path.join(csv_path, filename)
            
            # Read .dpt file and extract wavelength and intensity data
            data = np.genfromtxt(dpt_path, dtype=float, delimiter=",")
            data = data[data[:,0].argsort()]
            wavelength = data[:, 0]
            intensity = data[:, 1]
            
            # Clip wavelength values to range [400, 4000]
            # wavelength = np.clip(wavelength, 400, 4000)
            
            # Initialize array for interpolated intensity values
            interpolated_intensity = np.zeros((1,4000))
            # x=np.arange(400, 4000)
            # Interpolate intensity values
            # print(wavelength[(len(wavelength)-10):-1])
            # print(wavelength[:10])
            # print(intensity[:10])
            # print(intensity[-10:-1])
            for i in np.arange(400,4000):
                interpolated_intensity[0,i] = np.interp(i, wavelength, intensity)
            
            
            # Plot interpolated intensity values
            plt.plot(interpolated_intensity[0,:])
            plt.xlabel("Wavelength")
            plt.ylabel("Intensity")
            plt.title("Interpolated IR Spectrum")
            
            # Save the plot with the same filename
            save_path = os.path.splitext(dpt_path)[0] + ".png"
            plt.savefig(save_path)
            plt.close()
            if array is None:
                array = interpolated_intensity
            else:
                array = np.vstack((array, interpolated_intensity))
    save_path = os.path.join(csv_path, save_name)
    np.save(save_path, array)

def save_indexes_to_file(indexes, filename):
    with open(filename, 'w') as file:
        for index in indexes:
            file.write(str(index) + '\n')

def load_args_from_txt(txt_file, args):
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if line:
            arg_name, arg_value = line.split(': ')
            arg_name = arg_name.strip()
            arg_value = arg_value.strip()

            # Check the type of the argument and convert it accordingly
            if arg_name in args.__dict__:
                arg_type = type(args.__dict__[arg_name])
                if arg_type == int:
                    arg_value = int(arg_value)
                elif arg_type == float:
                    arg_value = float(arg_value)
                elif arg_type == bool:
                    arg_value = arg_value.lower() == 'true'

                setattr(args, arg_name, arg_value)



def plot_pca_scatter(data_list, save_path):
    from sklearn.decomposition import PCA
    # Step 2: Apply PCA
    colors = plt.cm.rainbow(np.linspace(0, 1, len(data_list)))
    for i, data in enumerate(data_list):   
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data)

        # Step 3: Plot the samples on a 2D scatter plot with index annotations
        plt.scatter(pca_result[:, 0], pca_result[:, 1], label=f'Array {i}', color=colors[i])

        # Annotate each point with its index
        for i, (x, y) in enumerate(zip(pca_result[:, 0], pca_result[:, 1])):
            plt.text(x, y, str(i), color='red', fontsize=8, ha='center', va='center')

        plt.title('2D Scatter Plot after PCA with Index Annotations')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.savefig(save_path)
if __name__=="__main__":
    # Plotting ==========================
    # x = np.load("data/open_specy/PP.npy")
    # name_all = np.load("data/real/FTIR_name.npy")
    # name_notPolymer = np.array([i for i in name_all if i.startswith("notPolymer")])
    # plot9Data(x, "plots/real/open_spcey/PP", all=True)

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
    # plt.savefig("check.png")
    # Baseline correction eddie ===================================
    # array = np.load("data/open_specy/from_eddie_raw/HDPE.npy")
    # asls_dataset(array, "data/open_specy/from_eddie_raw/HDPE_baseline.npy")
    # array = np.load("data/open_specy/from_eddie_raw/LDPE.npy")
    # asls_dataset(array, "data/open_specy/from_eddie_raw/LDPE_baseline.npy")
    # array = np.load("data/open_specy/from_eddie_raw/PET.npy")
    # asls_dataset(array, "data/open_specy/from_eddie_raw/PET_baseline.npy")
    # array = np.load("data/open_specy/from_eddie_raw/PP.npy")
    # asls_dataset(array, "data/open_specy/from_eddie_raw/PP_baseline.npy")
    # array = np.load("data/open_specy/from_eddie_raw/others.npy")
    # asls_dataset(array, "data/open_specy/from_eddie_raw/others_baseline.npy")
    
    data0 = np.load("/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_warwick/PP_new.npy")
    data1 = np.load("/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_warwick/PET.npy")
    plot_pca_scatter([data0, data1], save_path="check.png")
