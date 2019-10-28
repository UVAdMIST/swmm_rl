# swmm_rl
Repo for SWMM RL code

Push stuff into develop

## Notes

New organization

Currently has "run_swmm" section which is basically the stuff that is not yet modularized

rewards and evs are separated


The goal in my mind is to only interact with the main file while training and testing new agents.

main calls different environments and utilities to build a new network or re-use an existing one, sets up the appropriate gymm environment, chooses data, etc.

There are a lot of moving parts to this, so specifying this with command line arguments could get clunky fast.

For now, main should just call the appropriate "run_swmm" file, i.e. whatever file everybody has already been working on.

### Note on venv stuff
I'm not sure what's the best practice here. (Or least worst)
- track venv, packages, and python
- have packages installed in docker image, untracked. somehow link those packages to venv so that you install . as well
- just put -e . into requirements and don't worry about it since it's a docker image


## To Do

automate and separate neural network setup
- Check if that's already automated if you're feeling lucky/excited

Maybe use text files to store hyperparameter/weight preferences of people?

Fix file paths
- Support windows + unix
- localize paths inside the project
	- e.g. file paths should start inside the project's root, unless the paths are temporary. (command line arguments, setting files etc.)


## Installation

python3 -m venv venv
source ./venv/bin/activate
pip3 install -e .	# this could be added into requirements.txt later

pip3 install -r requirements.txt


