from utils.datasets import ComplexLoader, ClassMagLoader, ClassComplexLoader
from utils.autoMRF import AutoClassMRF, AutoRegMRF


train = ClassComplexLoader("/mikQNAP/augmented_data/MRF/*","/mikQNAP/augmented_data/T1w/*","/mikQNAP/augmented_data/T2w/*",5, num_classes=32)
test = ClassComplexLoader("/mikQNAP/augmented_data/MRF_test/*","/mikQNAP/augmented_data/T1w_test/*","/mikQNAP/augmented_data/T2w_test/*",1, num_classes=32)

AMRF = AutoClassMRF(256,10,4, num_classes=32)
AMRF.fit(train, test)
