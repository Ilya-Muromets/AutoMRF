from utils.datasets import *
from utils.autoMRF import *
import time
import torch


# train = ClassComplexLoader("data/MRF/*","data/T1w/*","data/T2w/*", num_classes=2**16)
# test = ClassComplexLoader("data/MRF_test/*","data/T1w_test/*","data/T2w_test/*", num_classes=2**16)

# AMRF = AutoRegMRF(batchsize=512, epochs=100, workers=8, model_name="regepoch100T1")
# AMRF.fit(train, test)

train = ClassComplexLoader("data/MRF/*","data/T2w/*","data/T1w/*", num_classes=2**16)
test = ClassComplexLoader("data/MRF_test/*","data/T2w_test/*","data/T1w_test/*", num_classes=2**16)

AMRF = AutoRegMRF(batchsize=512, epochs=100, workers=8, model_name="regL1")
AMRF.fit(train, test)
