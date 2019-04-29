import glob
import numpy as np

filenames = glob.glob("/mikQNAP/augmented_data/T2w/*")
min_thresh = 0
max_thresh = 0
for filename in filenames:
            # round to int within class number
            T1 = np.load(filename)
            # throw out outliers
            max_thresh += np.percentile(T1,99)
            min_thresh += np.min(T1)

min_thresh = min_thresh/len(filenames)
max_thresh = max_thresh/len(filenames)
print(min_thresh, max_thresh)
for filename in filenames:
            # round to int within class number
            T1 = np.load(filename)
            T1[T1>max_thresh]=max_thresh
            T1[T1<min_thresh]=min_thresh
            T1 = T1 - min_thresh
            T1 = T1/(max_thresh - min_thresh)
            T1 = np.round(((2**16-1))*T1, 0).astype(np.int)
            save_name = filename.split("/")[-1]
            np.save("data/T2w/" + save_name, T1.astype(np.uint16))

filenames = glob.glob("/mikQNAP/augmented_data/T2w_test/*")
for filename in filenames:
            # round to int within class number
            T1 = np.load(filename)
            T1[T1>max_thresh]=max_thresh
            T1[T1<min_thresh]=min_thresh
            T1 = T1 - min_thresh
            T1 = T1/(max_thresh - min_thresh)
            T1 = np.round(((2**16-1))*T1, 0).astype(np.int)
            save_name = filename.split("/")[-1]
            np.save("data/T2w_test/" + save_name, T1.astype(np.uint16))
