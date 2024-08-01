#!/bin/sh
env="Gridworld"
plan=1
algo="mappo_mgda" #"mappo" "ippo"
exp="check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_gridworld.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --plan ${plan} --seed ${seed} --hidden_size 64 --use_mgda --entropy_coef 0.01 --use_proper_timelimit \
    --n_training_threads 1 --n_rollout_threads 10 --num_mini_batch 1 --episode_length 200 --num_env_steps 500_000 \
    --ppo_epoch 5 --use_ReLU --gain 0.01 --lr 5e-4 --critic_lr 5e-4 --wandb_name "xxx" --user_name "_" --cuda --use_eval --deterministic_eval \
    --seperated_rewards
done