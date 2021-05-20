import argparse

from models.RISAN_catdog_wa import CatDogVgg as CatsvsDogDean
from models.RISAN_CIFAR import CIFARVgg as cifar10Dean
# from models.RISAN_mnist import Mnistvgg as mnistDEAN
from RISAN_utils_NA import *

MODELS = {"cifar_10": cifar10Dean, "catsdogs": CatsvsDogDean}

def to_train(filename):
    checkpoints = os.listdir("history_checkpoints/")
    if filename in checkpoints:
        return False
    else:
        return True

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar_10')

# parser.add_argument('--model_name', type=str, default='test')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--gamma',type=float, default=1e-3)
parser.add_argument('--filename',type=str,default='cifar10.h5')
parser.add_argument('--epochs',type=int,default=250)
parser.add_argument('--noise',type=float,default=0.0)
args = parser.parse_args()

model_cls = MODELS[args.dataset]
# model_name = args.model_name
# baseline_name = args.baseline

cost_reject = [ 0.4, 0.35, 0.3, 0.25]

for cost in cost_reject:
    # filename = args.dataset+str(cost)+'.h5'
    # catdog_model = CatsvsDogDean(Train,filename,alpha,gamma)
    model = model_cls(train=to_train("{}_{}.h5".format(args.dataset, cost)), filename="{}_{}.h5".format(args.dataset, cost), alpha=args.alpha,gamma= args.gamma,maxepochs=args.epochs,cost_rej=cost,noise_frac=args.noise)

