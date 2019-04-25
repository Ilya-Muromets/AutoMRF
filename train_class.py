from utils.datasets import *
from utils.autoMRF import *
import time
import torch

train = ClassComplexLoader("data/MRF/*","data/T1w/*","data/T2w/*", num_classes=256)
test = ClassComplexLoader("data/MRF_test/*","data/T1w_test/*","data/T2w_test/*", num_classes=256)

# MRF, T1, T2 = train[160*119]
# MRF = torch.from_numpy(MRF)
# print(torch.norm(Variable(MRF.type(torch.FloatTensor).cuda())))
# print(T1)
# AMRF = AutoClassMRF(batchsize=1024,epochs=20,workers=8, num_classes=4096, alpha=0)
# AMRF.fit(train, test)


# AMRF = AutoClassMRF(512,3,8, num_classes=256, alpha=0.0, model_name="batch1024class256epoch3alpha0.0")
# AMRF.fit(train, test)

# AMRF = AutoClassMRF(512,3,8, num_classes=256, alpha=0.25, model_name="batch1024class256epoch3alpha0.25")
# AMRF.fit(train, test)

# AMRF = AutoClassMRF(512,3,8, num_classes=256, alpha=0.5, model_name="batch1024class256epoch3alpha0.5")
# AMRF.fit(train, test)

# AMRF = AutoClassMRF(512,3,8, num_classes=256, alpha=0.75, model_name="batch1024class256epoch3alpha0.75")
# AMRF.fit(train, test)

AMRF = AutoClassMRF(512,3,8, num_classes=256, alpha=1.0, model_name="batch1024class256epoch3alpha1.0")
AMRF.fit(train, test)

AMRF = AutoClassMRF(512,3,8, num_classes=256, alpha=0.4, model_name="batch1024class256epoch3alpha0.4")
AMRF.fit(train, test)

AMRF = AutoClassMRF(512,3,8, num_classes=256, alpha=0.6, model_name="batch1024class256epoch3alpha0.6")
AMRF.fit(train, test)

train = ClassComplexLoader("data/MRF/*","data/T1w/*","data/T2w/*", num_classes=512)
test = ClassComplexLoader("data/MRF_test/*","data/T1w_test/*","data/T2w_test/*", num_classes=512)

AMRF = AutoRegMRF(batchsize=512,epochs=50,workers=8, num_classes=512, model_name="batch512class512epoch50reg")
AMRF.fit(train, test)
