from utils.datasets import *
from utils.autoMRF import *
import time
import torch

train = ClassComplexLoader("data/MRF/*","data/T2w/*","data/T1w/*", num_classes=1024)
test = ClassComplexLoader("data/MRF_test/*","data/T2w_test/*","data/T1w_test/*", num_classes=1024)

AMRF = AutoClassMRF(batchsize=512, epochs=50, workers=8, alpha=0.8, num_classes=1024, model_name="class1024alpha80epoch50T2")
AMRF.fit(train, test)

print("done class alpha 0.8")

train = ClassComplexLoader("data/MRF/*","data/T1w/*","data/T2w/*", num_classes=1024)
test = ClassComplexLoader("data/MRF_test/*","data/T1w_test/*","data/T2w_test/*", num_classes=1024)

AMRF = AutoClassMRF(batchsize=512, epochs=50, workers=8, alpha=0.2, num_classes=1024, model_name="class1024alpha20epoch50T1")
AMRF.fit(train, test)

train = ClassComplexLoader("data/MRF/*","data/T2w/*","data/T1w/*", num_classes=1024)
test = ClassComplexLoader("data/MRF_test/*","data/T2w_test/*","data/T1w_test/*", num_classes=1024)

AMRF = AutoClassMRF(batchsize=512, epochs=50, workers=8, alpha=0.2, num_classes=1024, model_name="class1024alpha20epoch50T2")
AMRF.fit(train, test)

print("done class alpha 0.8")

train = ClassComplexLoader("data/MRF/*","data/T1w/*","data/T2w/*", num_classes=2**16)
test = ClassComplexLoader("data/MRF_test/*","data/T1w_test/*","data/T2w_test/*", num_classes=2**16)

AMRF = AutoRegMRF(batchsize=512, epochs=100, workers=8, model_name="regepoch100T1")
AMRF.fit(train, test)

train = ClassComplexLoader("data/MRF/*","data/T2w/*","data/T1w/*", num_classes=2**16)
test = ClassComplexLoader("data/MRF_test/*","data/T2w_test/*","data/T1w_test/*", num_classes=2**16)

AMRF = AutoRegMRF(batchsize=512, epochs=100, workers=8, model_name="regepoch100T2")
AMRF.fit(train, test)
