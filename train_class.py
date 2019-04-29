from utils.datasets import *
from utils.autoMRF import *
import time
import torch

train = ClassComplexLoader("data/MRF/*","data/T1w/*","data/T2w/*", num_classes=2**16)
test = ClassComplexLoader("data/MRF_test/*","data/T1w_test/*","data/T2w_test/*", num_classes=2**16)

AMRF = AutoRegMRF(batchsize=700, epochs=20, workers=8, model_name="reg20epochMSEloss")
AMRF.fit(train, test)

print("Done long reg")

