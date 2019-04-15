from utils.datasets import *
from utils.autoMRF import *


train = ComplexLoader("/mikQNAP/augmented_data/MRF/*","/mikQNAP/augmented_data/T1w/*","/mikQNAP/augmented_data/T2w/*",5)
test = ComplexLoader("/mikQNAP/augmented_data/MRF_test/*","/mikQNAP/augmented_data/T1w_test/*","/mikQNAP/augmented_data/T2w_test/*",1)

AMRF = AutoRegMRF(256,50,4)
AMRF.fit(train, test)
