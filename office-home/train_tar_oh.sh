seed=2019

python train_tar.py --home --dset a2r  --K 3 --KK 2 --lr 5e-4 --batch_size 32 --seed $seed

python train_tar.py --home --dset r2a  --K 3 --KK 2 --lr 5e-4 --batch_size 32 --seed $seed

python train_tar.py --home --dset r2c  --K 3 --KK 2 --lr 5e-4 --batch_size 32 --seed $seed

python train_tar.py --home --dset r2p  --K 3 --KK 2 --lr 5e-4 --batch_size 32 --seed $seed

python train_tar.py --home --dset p2a  --K 3 --KK 2 --lr 5e-4 --batch_size 32 --seed $seed

python train_tar.py --home --dset p2c  --K 3 --KK 2 --lr 5e-4 --batch_size 32 --seed $seed

python train_tar.py --home --dset a2p  --K 3 --KK 2 --lr 5e-4 --batch_size 32 --seed $seed 

python train_tar.py --home --dset a2c  --K 3 --KK 2 --lr 5e-4 --batch_size 32 --seed $seed

python train_tar.py --home --dset p2r  --K 3 --KK 2 --lr 5e-4 --batch_size 32 --seed $seed

python train_tar.py --home --dset c2a  --K 3 --KK 2 --lr 5e-4 --batch_size 32 --seed $seed

python train_tar.py --home --dset c2p  --K 3 --KK 2 --lr 5e-4 --batch_size 32 --seed $seed

python train_tar.py --home --dset c2r  --K 3 --KK 2 --lr 5e-4 --batch_size 32 --seed $seed