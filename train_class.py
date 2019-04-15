from utils.datasets import ComplexLoader, ClassMagLoader
from utils.autoMRF import AutoClassMRF, AutoRegMRF


train = ClassMagLoader("MRF_magnitude/*","/mikQNAP/augmented_data/T1w/*","/mikQNAP/augmented_data/T2w/*",1, num_classes=16)
test = ClassMagLoader("MRF_test_magnitude/*","/mikQNAP/augmented_data/T1w_test/*","/mikQNAP/augmented_data/T2w_test/*",1, num_classes=16)

AMRF = AutoClassMRF(256,50,4, num_classes=16)
AMRF.fit(train, test)
