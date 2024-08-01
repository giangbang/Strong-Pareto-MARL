# Experiments with MARL environments

This part is based on the code from the paper "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" in NeurIPS 2022 ([link code](https://github.com/marlbenchmark/on-policy)).

### Tested environments
- `gridworld`: four scenarios `Door`, `Dead End`, `Two Corridors`, and `Two Rooms`.
- `MPE`: two scenarios `simple_reference` and `simple_spread`.

### Experiments
To run the experiments, `cd` to the appropriate script folders:

- `gridworld`: see scripts in the folder `train/train_gridworld_scripts`. 

- `MPE`: see scripts in the folder `train/train_mpe_scripts`

### Results
#### gridworld
![avatar](/onpolicy/assets/gridworld.png)
#### MPE
![avatar](/onpolicy/assets/mpe.png)