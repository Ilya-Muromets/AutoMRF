import glob
import numpy as np
file_list = glob.glob("/mikQNAP/augmented_data/MRF/*")
for file_name in file_list:
    x = np.load(file_name)
    x_mag = np.abs(x[0:500] + x[500:]*1j)
    np.save("MRF_magnitude/" + file_name.split("/")[-1], x_mag)
    print(file_name, "done")
    
file_list = glob.glob("/mikQNAP/augmented_data/MRF_test/*")
for file_name in file_list:
    x = np.load(file_name)
    x_mag = np.abs(x[0:500] + x[500:]*1j)
    np.save("MRF_test_magnitude/" + file_name.split("/")[-1], x_mag)
    print(file_name, "done")