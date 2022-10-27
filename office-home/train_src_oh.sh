seed=$1

python train_src.py --dset p2c --home --seed $seed

python train_src.py --dset p2r --home --seed $seed

python train_src.py --dset p2a --home --seed $seed

python train_src.py --dset a2p --home --seed $seed

python train_src.py --dset a2r --home --seed $seed

python train_src.py --dset a2c --home --seed $seed

python train_src.py --dset r2a --home --seed $seed

python train_src.py --dset r2p --home --seed $seed

python train_src.py --dset r2c --home --seed $seed

python train_src.py --dset c2r --home --seed $seed

python train_src.py --dset c2a --home --seed $seed

python train_src.py --dset c2p --home --seed $seed




