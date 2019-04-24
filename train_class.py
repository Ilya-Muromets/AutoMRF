from utils.datasets import *
from utils.autoMRF import *
import time

train = ClassComplexLoader("data/MRF/*","data/T1w/*","data/T2w/*", num_classes=256)
test = ClassComplexLoader("data/MRF_test/*","data/T1w_test/*","data/T2w_test/*", num_classes=256)

# train[11912356]
AMRF = AutoClassMRF(batchsize=1024,epochs=5,workers=32, num_classes=256, alpha=0.7)
AMRF.fit(train, test)


# AMRF = AutoClassMRF(1024,16,8, num_classes=256, alpha=0.0, model_name="batch1024class256epoch16alpha0.0")
# AMRF.fit(train, test)

# AMRF = AutoClassMRF(1024,16,8, num_classes=256, alpha=0.2, model_name="batch1024class256epoch16alpha0.2")
# AMRF.fit(train, test)

# AMRF = AutoClassMRF(1024,16,8, num_classes=256, alpha=0.4, model_name="batch1024class256epoch16alpha0.4")
# AMRF.fit(train, test)

# AMRF = AutoClassMRF(1024,16,8, num_classes=256, alpha=0.6, model_name="batch1024class256epoch16alpha0.6")
# AMRF.fit(train, test)

# AMRF = AutoClassMRF(1024,16,8, num_classes=256, alpha=0.8, model_name="batch1024class256epoch16alpha0.8")
# AMRF.fit(train, test)

# AMRF = AutoClassMRF(1024,16,8, num_classes=256, alpha=1.0, model_name="batch1024class256epoch16alpha1.0")
# AMRF.fit(train, test)
