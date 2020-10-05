# Some exemplary runs from the streaming experiments in the paper, sEM side
# all relevant parameters from the paper are given explicitly

# The ./ExpDist directory must exist!
mkdir ./ExpDist

# Full MNIST
python3 ../SGD_GMM/GMM.py --D1 1 2 3 4 5 6 7 8 --D2 0 --D3 9 --dataset_file MNIST --epochs 2 --exp_id 0_gmminc --L2_eps0 0.1 --L2_epsInf 0.001 --L2_regularizer_delta 0.05 --L2_regularizer_reset_eps 0.001 --L2_regularizer_reset_sigma 0.01 --L2_sigmaUpperBound 20.0 --measuring_points 10 --muInit 0.1 --noise 0.0 --slice -1 -1 --tmp_dir ./ExpDist

# 6x6-Patch MNIST
python3 ../SGD_GMM/GMM.py --D1 1 2 3 4 5 6 7 8 --D2 0 --D3 9 --dataset_file MNIST --epochs 2 --exp_id 0_gmminc --L2_eps0 0.1 --L2_epsInf 0.001 --L2_regularizer_delta 0.05 --L2_regularizer_reset_eps 0.001 --L2_regularizer_reset_sigma 0.01 --L2_sigmaUpperBound 20.0 --measuring_points 10 --muInit 0.1 --noise 0.0 --slice 6 6 --tmp_dir ./ExpDist
