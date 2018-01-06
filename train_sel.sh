python3.6 train.py ~/DATASETS/cifar.python cifar10 -s ./snapshots/SelNext --log ./logs/SelNext --model SelNext --cardinality 32 --base_width 16 --ngpu 1 --learning_rate 0.015 -b 32

python3.6 train.py ~/DATASETS/cifar.python cifar10 -s ./snapshots/DropNext --log ./logs/DropNext --model DropNext --cardinality 32 --base_width 16 --ngpu 1 --learning_rate 0.015 -b 32

python3.6 train.py ~/DATASETS/cifar.python cifar10 -s ./snapshots/Res164 --log ./logs/Res164 --model ResNext --cardinality 1 --base_width 32 --depth 164 --ngpu 1 --learning_rate 0.03 -b 64

python3.6 train.py ~/DATASETS/cifar.python cifar10 -s ./snapshots/DropCombine -l ./snapshots/DropCombine/model.pytorch --log ./logs/DropCombine --model DropCombine --cardinality 1 --base_width 64 --depth 164 --ngpu 1 --learning_rate  0.01 -b 64

python3.6 train.py ~/DATASETS/cifar.python cifar10 -s ./snapshots/ResNext164 --log ./logs/ResNext164 --model ResNext --cardinality 4 --base_width 16 --depth 164 --ngpu 1 --learning_rate 0.03 -b 64

python3.6 train.py ~/DATASETS/cifar.python cifar10 -s ./snapshots/ResNext164__lr0.1 --log ./logs/ResNext164__lr0.1 --model ResNext --cardinality 1 --base_width 16 --band_width 16 --depth 164 --ngpu 1 --learning_rate 0.1 -b 64

python3.6 train.py ~/DATASETS/cifar.python cifar10 -s ./snapshots/Preact_ResNext164 --log ./logs/Preact_ResNext164 --model ResNext --cardinality 1 --base_width 16 --band_width 16 --depth 164 --ngpu 1 --learning_rate 0.1 -b 64

python3.6 train.py ~/DATASETS/cifar.python cifar10 -s ./snapshots/ResNext164lr0.1 --log ./logs/ResNext164lr0.1 --model ResNext --cardinality 1 --base_width 16 --band_width 16 --depth 164 --ngpu 1 --learning_rate 0.1 -b 64

python3.6 train.py ~/DATASETS/cifar.python cifar10 -s ./snapshots/PreAResNext164lr0.1 --log ./logs/PreAResNext164lr0.1 --model ResNext --cardinality 1 --base_width 16 --band_width 16 --depth 164 --ngpu 1 --learning_rate 0.1 -b 64 --preact

python3.6 train.py ~/DATASETS/cifar.python cifar10 -s ./snapshots/DropcombinePreact --log ./logs/DropcombinePreact --model DropCombine --cardinality 1 --base_width 16 --band_width 16 --depth 164 --ngpu 1 --learning_rate 0.1 -b 64 --preact

python3.6 train.py ~/DATASETS/cifar.python cifar10 -s ./snapshots/Dropcombine --log ./logs/Dropcombine --model DropCombine --cardinality 1 --base_width 16 --band_width 16 --depth 164 --ngpu 1 --learning_rate 0.1 -b 64