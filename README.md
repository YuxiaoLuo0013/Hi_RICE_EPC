# Architecting Urban Epidemic Defense: A Hierarchical Region-Individual Control Framework for Optimizing Large-Scale Individual Mobility Interventions

## Project Overview

This project aims to develop a hierarchical region-individual control framework to optimize large-scale individual mobility interventions, effectively defending against urban epidemics. By integrating control strategies at both regional and individual levels, we seek to minimize infection transmission and socioeconomic costs.

## Installation

To set up the project environment, you'll need Python 3.9+ and pip installed on your system. Follow these steps to install the required dependencies:

Install the required packages:

```
pip install -r requirements.txt
```

## Simulation Environment Setting

### Agent-based Model Simulation @EPC_env

This project implements an advanced Agent-based Model (ABM) for simulating epidemic spread in urban environments. The `EpidemicSimulation` class, located in `EpidemicSimulation1.py`, forms the core of this simulation. It models various aspects of epidemic dynamics, including disease compartments (e.g., susceptible, latent, infectious), vaccination strategies, and public health interventions. The simulation incorporates detailed population demographics, spatial information, and complex interaction networks (home, work, school, community). Key features include the ability to simulate different vaccination uptake scenarios, implement various non-pharmaceutical interventions (NPIs), and model the impact of drug treatments. The simulation also accounts for time-dependent factors such as immunity waning and reinfection possibilities. 

Advanced risk assessment methods, including a novel Infectious-risk Assessment Model (I-RAM), are implemented to calculate infection probabilities based on contact patterns and intervention measures. This comprehensive ABM provides a powerful tool for analyzing and optimizing epidemic control strategies in large-scale urban settings. Infectious-risk Assessment Model (I-RAM) is set in `get_situation_features` function.

### Simulation Setting file

Files  /Parameters and /Property/ are the data used for simulation environment setting includding the synthetic population data, and can be download from https://pan.baidu.com/s/1mKXr57qq-KTNkCo8JGvJKA (code: 6892). 
You need to put them in the right location /env/EPC_env/.

### Reinforcement Learning Simulation @env_ma.py

EpidemicModel class: The main environment class that inherits from gym.Env.
Initializes the epidemic simulation using EpidemicSimulation from EpidemicSimulation1.py.
Defines observation and action spaces.
Implements reset(), step(), and get_reward() methods.

## Run training

train.py implements a reinforcement learning (RL) approach to epidemic control in urban settings. It consists of a custom OpenAI Gym environment and a PPO (Proximal Policy Optimization) agent for learning optimal intervention strategies.

And you can open fold Hi_RICE and use run.sh to run the training.

## Run Testing

After training, you can use  to run the testing by record_daily_Hirice.py.
These is some parameters you need to set before running the testing:

1. `seed`: Random seed for reproducibility (default: 10000)
2. `num_envs`: Number of parallel environments to run (default: 10)
3. `p_trans`: Transmission probability of the disease (default: 0.07)
4. `use_gpu`: Boolean flag to use GPU if available (default: True)
5. `n_pop`: Total population size (default: 1618304)
6. `n_days`: Number of days to simulate (default: 150)
7. `mean`: Path to the .npy file containing the mean values for observation normalization
8. `variance`: Path to the .npy file containing the variance values for observation normalization
9. `model_path`: Path to the trained model file (.pth)

## Result Analysis

After running the testing script, you'll get the following results and outputs:

1. Console Output:
   
   - Mean total infected population
   - Mean total quarantined population
   - Final score (calculated based on infections and quarantines)

2. Saved Data:
   The script can save the following data (uncomment the relevant lines in the script to enable saving):
   
   - Daily infection counts: `./daily_record/{p_trans}_daily_I.npy`
   - Daily quarantine counts: `./daily_record/{p_trans}_daily_Q.npy`
   - Regional quarantine logs: `./daily_record/{p_trans}_Region_Q.npy`

These data can be used to analyze the performance of the trained model in controlling the epidemic, including its effectiveness in reducing infections and managing quarantines across different regions and over time.

We provide a visualization.ipynb for guiding researchers to reproduce findings in our paper. 