# Some exemplary runs from the streaming experiments in the paper, sEM side
# all relevant parameters from the paper are given explicitly

# The ./ExpDist directory must exist!
mkdir ./ExpDist

# Full MNIST
python3 ../SGD_GMM/emAlgos.py --alpha 0.01 --alpha0 0.1 --D1 1 2 3 4 5 6 7 8 --D2 0 --D3 9 --DAll 0 1 2 3 4 5 6 7 8 9 --dataset_file MNIST --exp_id 0_seminc --initMode random --initMus 0.1 --initPrecs 20 --mode sEM --n 8 --nrTasks 3 --nrTestSamples 1000 --rhoMin 0.001 --slice -1 -1 --taskEpochs 2 2 2 --tmp_dir ./ExpDist

# 6x6-Patch MNIST
python3 ../SGD_GMM/emAlgos.py --alpha 0.01 --alpha0 0.1 --D1 1 2 3 4 5 6 7 8 --D2 0 --D3 9 --DAll 0 1 2 3 4 5 6 7 8 9 --dataset_file MNIST --exp_id 0_seminc --initMode random --initMus 0.1 --initPrecs 20 --mode sEM --n 8 --nrTasks 3 --nrTestSamples 1000 --rhoMin 0.001 --slice 6 6 --taskEpochs 2 2 2 --tmp_dir ./ExpDist
