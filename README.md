# PDEGPT
The data and codes for "Generative Discovery of Partial differential Equation by Learning from Math Handbooks"

# Environments
python==3.12.4   
pandas==2.2.2  
numpy==1.26.4 
pytorch==2.3.1 
scikit-learn==1.4.2  
scipy==1.13.1 
matplotlib==3.8.4


# The dataset utilized in this work 
* PDE_dataset_0725.xlsx (The dataset of PDE forms collected from the math handbook, "Nonlinear Partial Differential Equations for Scientists and Engineers", 221 PDEs)
* The solutions of dynamic systems for proof-of-concept experiments, inlucing:
  - Korteweg-De Vries (KdV) equation
  - Burgers’ equation
  - Convection-diffusion equation
  - Wave equation
  - Chaffee-infante equation
  - Klein-Gordon (KG) equation
  - Eq. (6.2.12) in the handbook
  - Eq. (8.14.d) in the handbook
  - Allen-Cahn equation
  - 2D Burgers equation
  - Poisson's equation solved in disk region
  - Poisson's equation solved in smliery face region
  - Poisson's equation solved in "EITech" region
  - Poisson's equation solved in 3D space shuffle region (too big, provide in figshare)
  - The simulation of the vibration of an H-shaped elastic membrane (too big, provide in figshare)

# How to reproduce our work?
1. Install relevant packages (~20 min)  
2. Run the surrogate_model.py (about 2 min for training the EqGPT model and 1~10 min {according to the data_num} for training the surrogate model from sparse and noisy data)
   This work utilize the GPU acceleration by Nvidia Gefore GTX 4090D  
3. Run the continue_train_GPT.py to fine-tune the EqGPT with the rewards of generated PDEs calculated through meta-data and discover the optimal PDE structure (2~5 min).
4. Run the PINN_optimization.py to further optimize the coefficients of discovered PDEs via PINN (~1 min).
* Note: the continue_train_GPT.py is controlled by the parameters in surrogate_model.py (i.e., Equation_name, data_num, noise level, etc)  


# Expected outputs
* All outputs will be saved in the dir model_save, gpt_model, and result_save  
* Surrogate models of the data are saved in dir of "model_save". (These models can be used for generate meta-data and calculate derivatives)
* EqGPT models are saved in the dir of "gpt_model".
* The discovered equations and corresponding awards are saved in "result_save", including awards.csv, equations.csv, and sentences.pkl, which records the best 10 PDE structures in each optimization epoch.
  (An example of results is provided in this repository for discovery of KdV equation under 10000 data and 50% Gaussian noise)
