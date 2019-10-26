from utils.datasets import ClassComplexLoader
from utils.autoMRF import *
import time
import torch
import argparse

parser = argparse.ArgumentParser(description='Trains model for MRF to T1/T2 contrast simulation.')
parser.add_argument('--model', type=str, default='div',  help='Model type. [reg | class | div]')
parser.add_argument('--model_name', type=str, default='test_name')
parser.add_argument('--num_classes', type=int, default=2**16, help='Quantization levels for T1/T2.')
parser.add_argument('--device', type=int, default=0, help="GPU")
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=512)
parser.add_argument('--workers', type=int, default=8)
args = parser.parse_args()

if args.model not in ['reg', 'class', 'div']:
    raise Exception("Model not valid.")

if args.model == 'reg':
    train = ClassComplexLoader("data/MRF/*","data/T1w/*","data/T2w/*", num_classes=args.num_classes)
    test = ClassComplexLoader("data/MRF_test/*","data/T1w_test/*","data/T2w_test/*", num_classes=args.num_classes)

    AMRF = AutoRegMRF(batchsize=args.batchsize, epochs=args.epochs, workers=args.workers, model_name=args.model_name, device=args.device)
    AMRF.fit(train, test)

# train = ClassComplexLoader("data/MRF/*","data/T1w/*","data/T2w/*", num_classes=2**16)
# test = ClassComplexLoader("data/MRF_test/*","data/T1w_test/*","data/T2w_test/*", num_classes=2**16)

# AMRF = AutoDivMRF(batchsize=512, epochs=100, workers=8, model_name="regepoch100T1", device=3)
# AMRF.fit(train, test)

# train = ClassComplexLoader("data/MRF/*","data/T1w/*","data/T1w/*", num_classes=2**16)
# test = ClassComplexLoader("data/MRF_test/*","data/T1w_test/*","data/T1w_test/*", num_classes=2**16)

# AMRF = AutoRegMRF(batchsize=512, epochs=200, workers=8, model_name="regL1T1")
# AMRF.fit(train, test)
