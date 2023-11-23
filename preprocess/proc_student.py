import os
import numpy as np
import matplotlib.pyplot as plt
import pybaselines

baseline = True
# Directory path containing .dpt files
directory = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/Data_Edward/student/Data/FTIR/"

# Loop through all .dpt files
array = None
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        dpt_path = os.path.join(directory, filename)
        
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
save_path = os.path.join(directory, "others1.npy")
print(array.shape)
np.save(save_path, array)    
