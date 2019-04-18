from utils.datasets import ComplexLoader, ClassMagLoader, ClassComplexLoader
from utils.autoMRF import AutoClassMRF, AutoRegMRF


train = ClassComplexLoader("/mikQNAP/augmented_data/MRF/*","/mikQNAP/augmented_data/T1w/*","/mikQNAP/augmented_data/T2w/*",5, num_classes=256)
test = ClassComplexLoader("/mikQNAP/augmented_data/MRF_test/*","/mikQNAP/augmented_data/T1w_test/*","/mikQNAP/augmented_data/T2w_test/*",1, num_classes=256)

alphas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
for alpha in alphas:
    AMRF = AutoClassMRF(256,5,1, num_classes=256, alpha=0.1, model_name="alpha" + str(alpha))
    AMRF.fit(train, test)
