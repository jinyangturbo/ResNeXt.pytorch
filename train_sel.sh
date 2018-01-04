python3.6 train.py ~/DATASETS/cifar.python cifar10 -s ./snapshots/SelNext --log ./logs/SelNext --model SelNext --cardinality 32 --base_width 16 --ngpu 1 --learning_rate 0.015 -b 32

python3.6 train.py ~/DATASETS/cifar.python cifar10 -s ./snapshots/DropNext --log ./logs/DropNext --model DropNext --cardinality 32 --base_width 16 --ngpu 1 --learning_rate 0.015 -b 32

python3.6 train.py ~/DATASETS/cifar.python cifar10 -s ./snapshots/DropCombine --log ./logs/DropCombine --model DropCombine --cardinality 1 --base_width 64 ---depth 164 -ngpu 1 --learning_rate 0.03 -b 64


