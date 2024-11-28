#!/bin/bash -l

#SBATCH --ntasks=110
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000
#SBATCH --time=12:00:00
#SBATCH --mail-user jan.dohmen@tuhh.de
#SBATCH --mail-type=ERR,FAIL

export OMP_NUM_THREADS=1
export WANDB__SERVICE_WAIT=300

# slurm loading
. /etc/profile.d/module.sh
module purge
module load anaconda/2020.11;

# conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "dynamicsswitch"

# cp work directory and change into it
WORKDIR=/work/dsf/cjd7625/
mkdir -p $WORKDIR/Repos
cp -aruf $HOME/Repos/aerl $WORKDIR/Repos
cd $WORKDIR/Repos/aerl/

cd experiments/jaxrl3/sequential/
# experiment config
n_runs=5
start_seed=0
project_name="DynamicsSwitch-sequential-2024-0"

static_args="--wandb --wandb_offline --project_name $project_name"
# static_args="--wandb --project_name $project_name"
changing_args=()

# 21
for env_name in "cartpole-swingup-v0"; do
  for timesteps in 1000000; do
    for exp_type in "basic"; do 
      for wrapper_id in "ZeroAction"; do
        for wrapper_dim in 0; do
          for wrapper_repeat in 4 20; do
            for wrapper_value in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
              changing_args+=("--env_name $env_name --timesteps $timesteps --exp_type $exp_type --wrapper_id $wrapper_id --wrapper_dim $wrapper_dim --wrapper_value $wrapper_value --wrapper_repeat $wrapper_repeat")
            done
          done
        done
      done
    done
  done
done

# run experiment
for ca in "${changing_args[@]}"
do
  for i in $(seq $n_runs)
  do
    seed=$((start_seed + i - 1))
    # echo python sac.py $static_args $ca
    srun --exclusive -N 1 -n 1 python sac.py --seed=$seed $static_args $ca &
    sleep 2
  done
done
wait

