from utils.datasets import ComplexLoader, ClassMagLoader
from utils.autoMRF import AutoMRF


train = ClassMagLoader("MRF_magnitude/*","/mikQNAP/augmented_data/T1w/*","/mikQNAP/augmented_data/T2w/*",10)
test = ClassMagLoader("MRF_test_magnitude/*","/mikQNAP/augmented_data/T1w_test/*","/mikQNAP/augmented_data/T2w_test/*",2)

AMRF = AutoMRF(128,50,4)
AMRF.fit(train, test)
