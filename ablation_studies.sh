CUDA_VISIBLE_DEVICES=0 python3 team_code/train.py --epochs 50 --batch_size 16 --save_program_files --lambda1 0.7
CUDA_VISIBLE_DEVICES=0 python3 team_code/train.py --epochs 50 --batch_size 16 --save_program_files --lambda1 0.3
CUDA_VISIBLE_DEVICES=1 python3 team_code/train.py --epochs 50 --batch_size 16 --save_program_files --lambda2 0.05 
CUDA_VISIBLE_DEVICES=1 python3 team_code/train.py --epochs 50 --batch_size 16 --save_program_files --lambda2 0.005
CUDA_VISIBLE_DEVICES=2 python3 team_code/train.py --epochs 50 --batch_size 16 --save_program_files --lambda3 0.7 <--
CUDA_VISIBLE_DEVICES=2 python3 team_code/train.py --epochs 50 --batch_size 16 --save_program_files --lambda3 0.3
