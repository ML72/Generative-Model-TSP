#!/bin/bash

# Training commands
echo "TRAINING TSP_50 MODEL 1: UNIFORM BASELINE"
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution unif --run_name baseline_unif_1
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution unif --run_name baseline_unif_2
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution unif --run_name baseline_unif_3
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution unif --run_name baseline_unif_4
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution unif --run_name baseline_unif_5

echo "TRAINING TSP_50 MODEL 2: HAC BASELINE"
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution unif --hardness_adaptive_percent 100 --run_name baseline_hac_1
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution unif --hardness_adaptive_percent 100 --run_name baseline_hac_2
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution unif --hardness_adaptive_percent 100 --run_name baseline_hac_3
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution unif --hardness_adaptive_percent 100 --run_name baseline_hac_4
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution unif --hardness_adaptive_percent 100 --run_name baseline_hac_5

echo "TRAINING TSP_50 MODEL 3: ABLATION NO VAE"
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution clusters --hardness_adaptive_percent 100 --run_name ablation_no_vae_1
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution clusters --hardness_adaptive_percent 100 --run_name ablation_no_vae_1
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution clusters --hardness_adaptive_percent 100 --run_name ablation_no_vae_1
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution clusters --hardness_adaptive_percent 100 --run_name ablation_no_vae_1
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution clusters --hardness_adaptive_percent 100 --run_name ablation_no_vae_1

echo "TRAINING TSP_50 MODEL 4: ABLATION NO HAC"
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution vae_curriculum --run_name ablation_no_hac_1
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution vae_curriculum --run_name ablation_no_hac_1
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution vae_curriculum --run_name ablation_no_hac_1
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution vae_curriculum --run_name ablation_no_hac_1
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution vae_curriculum --run_name ablation_no_hac_1

echo "TRAINING TSP_50 MODEL 5: VAE"
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution vae --hardness_adaptive_percent 100 --run_name vae_1
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution vae --hardness_adaptive_percent 100 --run_name vae_2
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution vae --hardness_adaptive_percent 100 --run_name vae_3
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution vae --hardness_adaptive_percent 100 --run_name vae_4
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --training_distribution vae --hardness_adaptive_percent 100 --run_name vae_5
