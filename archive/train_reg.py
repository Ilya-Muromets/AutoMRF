from utils.datasets import *
from utils.autoMRF import *


train = ComplexLoader("/mikQNAP/augmented_data/MRF/*","/mikQNAP/augmented_data/T1w/*","/mikQNAP/augmented_data/T2w/*",20)
test = ComplexLoader("/mikQNAP/augmented_data/MRF_test/*","/mikQNAP/augmented_data/T1w_test/*","/mikQNAP/augmented_data/T2w_test/*",4)

AMRF = AutoRegMRF(256,10,4)
AMRF.fit(train, test)
