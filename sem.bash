# Some exemplary runs from the streaming experiments in the paper, sEM side
# all relevant parameters from the paper are given explicitly

# The ./ExpDist directory must exist!
mkdir ./ExpDist

python3 ../SGD_GMM/emAlgos.py --alpha 0.01 --alpha0 0.05 --D1 0 1 2 3 4 5 6 7 8 9 --DAll 0 1 2 3 4 5 6 7 8 9 --dataset_file MNIST --exp_id 0_semstr --initMode random --initMus 0.1 --initPrecs 20 --mode sEM --n 8 --nrTasks 1 --nrTestSamples 1000 --rhoMin 0.001 --slice -1 -1 --taskEpochs 3 --tmp_dir ./ExpDist

python3 ../SGD_GMM/emAlgos.py --alpha 0.01 --alpha0 0.05 --D1 0 1 2 3 4 5 6 7 8 9 --DAll 0 1 2 3 4 5 6 7 8 9 --dataset_file MNIST --exp_id 30_semstr --initMode random --initMus 0.1 --initPrecs 20 --mode sEM --n 8 --nrTasks 1 --nrTestSamples 1000 --rhoMin 0.0001 --slice 6 6 --taskEpochs 3 --tmp_dir ./ExpDist

python3 ../SGD_GMM/emAlgos.py --alpha 0.01 --alpha0 0.1 --D1 0 1 2 3 4 5 6 7 8 9 --DAll 0 1 2 3 4 5 6 7 8 9 --dataset_file MNIST --exp_id 280_semstr --initMode random --initMus 0.1 --initPrecs 20 --mode sEM --n 8 --nrTasks 1 --nrTestSamples 1000 --rhoMin 0.001 --slice -1 -1 --taskEpochs 3 --tmp_dir ./ExpDist

python3 ../SGD_GMM/emAlgos.py --alpha 0.5 --alpha0 0.05 --D1 0 1 2 3 4 5 6 7 8 9 --DAll 0 1 2 3 4 5 6 7 8 9 --dataset_file MNIST --exp_id 560_semstr --initMode random --initMus 0.1 --initPrecs 20 --mode sEM --n 8 --nrTasks 1 --nrTestSamples 1000 --rhoMin 0.001 --slice -1 -1 --taskEpochs 3 --tmp_dir ./ExpDist

python3 ../SGD_GMM/emAlgos.py --alpha 0.5 --alpha0 0.1 --D1 0 1 2 3 4 5 6 7 8 9 --DAll 0 1 2 3 4 5 6 7 8 9 --dataset_file MNIST --exp_id 879_semstr --initMode random --initMus 0.1 --initPrecs 20 --mode sEM --n 8 --nrTasks 1 --nrTestSamples 1000 --rhoMin 0.0001 --slice 6 6 --taskEpochs 3 --tmp_dir ./ExpDist
