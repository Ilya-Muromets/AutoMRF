from utils.datasets import ComplexLoader, ClassMagLoader, ClassComplexLoader
from utils.autoMRF import AutoClassMRF, AutoRegMRF


train = ClassComplexLoader("/mikQNAP/augmented_data/MRF/*","/mikQNAP/augmented_data/T1w/*","/mikQNAP/augmented_data/T2w/*",15, num_classes=256)
test = ClassComplexLoader("/mikQNAP/augmented_data/MRF_test/*","/mikQNAP/augmented_data/T1w_test/*","/mikQNAP/augmented_data/T2w_test/*",3, num_classes=256)

AMRF = AutoClassMRF(1024,10,8, num_classes=256, alpha=0.375)
AMRF.fit(train, test)

AMRF = AutoClassMRF(1024,10,8, num_classes=256, alpha=0.625)
AMRF.fit(train, test)

