from utils.datasets import ComplexLoader, ClassMagLoader
from utils.autoMRF import AutoMRF


train = ClassMagLoader("MRF_magnitude/*","T1w/*","T2w/*",1)
test = ClassMagLoader("MRF_test_magnitude/*","T1w_test/*","T2w_test/*",1)


AMRF = AutoMRF(128,50,4)
AMRF.fit(train, test)
