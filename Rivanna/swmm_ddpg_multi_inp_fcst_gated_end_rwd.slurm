#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 2-23:55:00
#SBATCH -p gpu
#SBATCH --gres=gpu:p100:1
#SBATCH -A uvahydroinformatics
#SBATCH -o swmm_ddpg_multi_inp_fcst_gated_end_rwd.out
#SBATCH -e swmm_ddpg_multi_inp_fcst_gated_end_rwd.err

module purge
module load singularity tensorflow/1.12.0-py36

export SINGULARITYENV_MPLBACKEND="agg"
singularity-gpu exec /home/$USER/tensorflow-1.12.0-py36.simg python /home/$USER/swmm_rl/swmm_ddpg_multi_inp_fcst_gated_end_rwd.py
