from utils.datasets import ComplexLoader, ClassMagLoader
from utils.autoMRF import AutoMRF


train = ClassMagLoader("MRF_magnitude/*","/mikQNAP/augmented_data/T1w/*","/mikQNAP/augmented_data/T2w/*",5, num_classes=16)
test = ClassMagLoader("MRF_test_magnitude/*","/mikQNAP/augmented_data/T1w_test/*","/mikQNAP/augmented_data/T2w_test/*",1, num_classes=16)

AMRF = AutoMRF(128,50,4, num_classes=16)
AMRF.fit(train, test)
