 # Sample run command for RISAN
 Run on Cifar10 dataset-> python3 Train_DEAN.py --dataset cifar_10 --alpha 0.5 --gamma 1e-3 --epochs 250
 Run on Cifar10 dataset-> python3 Train_DEAN.py --dataset catsdogs --alpha 0.5 --gamma 1e-3 --epochs 250
 # Sample run command for RISAN-NA
 Run on Cifar10 dataset-> python3 Train_DEAN_NA.py --dataset cifar_10 --alpha 0.5 --gamma 1e-3 --epochs 250
 Run on Cifar10 dataset-> python3 Train_DEAN_NA.py --dataset catsdogs --alpha 0.5 --gamma 1e-3 --epochs 250
 # Values of cost of rejection (d) can be set in Train_DEAN File
 # A .npz file can be created for cats vs dogs dataset using DEAN_utils.py file