from utils.datasets import ComplexLoader
from utils.autoMRF import AutoMRF


train = ComplexLoader("/mikQNAP/augmented_data/MRF/*","/mikQNAP/augmented_data/T1w/*","/mikQNAP/augmented_data/T2w/*",10)
test = ComplexLoader("/mikQNAP/augmented_data/MRF_test/*","/mikQNAP/augmented_data/T1w_test/*","/mikQNAP/augmented_data/T2w_test/*",2)

AMRF = AutoMRF(128,50,4)
AMRF.fit(train, test)
