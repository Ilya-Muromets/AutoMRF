import numpy as np
import glob

T1_counts = np.zeros(2**16)
for filename in glob.glob("T1w/*.npy"):
    T1_counts += np.bincount(np.load(filename).flatten(), minlength=2**16)
np.save("T1_class_counts", T1_counts)

T2_counts = np.zeros(2**16)
for filename in glob.glob("T2w/*.npy"):
    T2_counts += np.bincount(np.load(filename).flatten(), minlength=2**16)
np.save("T2_class_counts", T2_counts)

