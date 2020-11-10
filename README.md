# swmm_rl
This repository contains Python codes for using reinforcement learning with the U.S EPA's Stormwater Management Model (SWMM) to develop real-time control policies of stormwater systems. Passive and Rule-based Control codes are included for comparison with RL. SWMM input files and forecast data are available on HydroShare: http://www.hydroshare.org/resource/e2d21c9224ab4aefaf1a5b6394b270b1.

This work has been published in the Journal of Hydroinformatics and is available via open access at https://iwaponline.com/jh/article/doi/10.2166/hydro.2020.080/77759/Flood-mitigation-in-coastal-urban-catchments-using

Required packages:
1. pyswmm
2. keras-rl (once installed, replace rl.core with modified file: core.py in this repo)
3. openai-gym

Codes for creating an RL environment from a SWMM input file and running RL are in the DDPG_Obs_Fcst folder. Weights for initializing the DDPG agent are included in that folder.

Rule-based control of SWMM simulations can be performed with code in the RuleBasedControl folder.

Passive (uncontrolled) SWMM simulations use code in the Passive folder.
