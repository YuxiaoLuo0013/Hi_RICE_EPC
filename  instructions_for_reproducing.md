This is the instructions for reproducing the results of the paper "Architecting Urban Epidemic Defense: A Hierarchical Region-Individual Control Framework for Optimizing Large-Scale Individual Mobility Interventions".
First you need download ABM data from the link in the README.md file.

##Section 5.2 Main Results

For this section, you can run the scripts run.sh to reproduce the results of Hi-RICE, can for the 3 R0-level scenarios you can adjust the parameters p_trans=0.07,0.15,0.24 for R0=2.5 4.5 6.5 in the script. After running the script, you can get the metrics in the logs file.

For other baselines, you can run the scripts in the folder "baselines" to reproduce the results, aslo change the parameters p_trans in the script or run.py file.

##Section 5.3 Ablation Study
For the ablation study, you first need to open ablation folder, and run the scripts in the scripts in each ablation version's folder to get the results (Individual control can be directly run by using the record_daily_individual.py). And you use the _Region_Q.npy file to draw the figure in the paper, the visualization code is in the ablation/visualization.ipynb file.

##Section 5.4 Investigation of Hyperparameter Impact
For this section, you can run the scripts in the Hi_RICE folder to get the results, and change the parameters 
default:
alpha=2.5
wta=1
wtp=1
to get different results. And the also you can get the figure from visualization.ipynb file.

##Section 5.5 Generalizability Analysis

For Assessing Generalizability with R 0Uncertainty you can run the record_daily_Hirice.py after you training the model, then you need to setiing the model parameter as the testing phase in README.md. You can easily change the parameters p_trans=0.07,0.15,0.24 for R0=2.5 4.5 6.5 to p_trans=0.04,0.11（2.5+-1）,0.12，0.18(4.5+-1), 0.2,0.25(6.5+-1) in this py file, finally you can get the results.

For Assessing Generalizability with Various Compliance Levels you can run the record_daily_Hirice.py after you training the model, then you need to setiing the model parameter as the testing phase in README.md. You can easily change the parameters mask_ratio=0.001，0.01，0.1 for different level in this py file, finally you can get the results.

##Section 5.6 Visualization of The Intervention Results
After training the model, you can use the results from folder daily_record as README.md to visualize the intervention results in the visualization.ipynb file.
