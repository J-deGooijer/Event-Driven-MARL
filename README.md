# Robust Event-Driven Interactions in Cooperative Multi-Agent Learning
Authors: Daniel Jarne Ornia, Manuel Mazo Jr
[Link to paper](https://arxiv.org/abs/2204.03361)

The code for the simulations is written in Python on its entirety. The authors used Anaconda for installation.
1. To install Anaconda, follow the instructions in https://docs.anaconda.com/anaconda/install/
2. Once Anaconda is installed, open a terminal window, navigate to S7 AE Code and install a conda envi- ronment from the file formats.yml (see https://docs.conda.io/projects/conda/en/latest/user-guide/ tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).
3. Activate the formats environment (https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/ manage-environments.html#activating-an-environment).

## Training a Q Function for a Particle Tag game
We can first learn a Q-Function (and policy) for a specific particle tag game. Using the command:
```
python train_Q.py --preQ path/to/existing/Q.pickle --episodes 100000
```
The loading of an existing Q function (optional) is in case we want to continue training an existing model. The episodes flag determines the learning epochs.
The rest of the parameters (learning rate, discount rate, environment size and number of agents) are hard-coded on the script file, fixed to the default values used in the paper. They can be modified for testing other combinations. For testing purposes, we set the environment size to a 5 × 5 arena (the learning is significantly faster), but note that the paper example is computed for a 10 × 10 environment, with 2 agents.
The script produces a pickle file containing the resulting Q table, in the folder /Qs/.
## Learning an SVR representation of Γα
To learn an SVR model for the robustness surrogate function, one can run:
```
python compute_robust_svr.py --preQ path/to/existing/Q.pickle --alpha 0.5 --nu 0.01 --C 100 --gamma 1
```
1. The parameter gamma is an SVR parameter (not related to the robustness surrogate function).
2. The script will produce a pickle file in the folder robustness surrogates containing the parameters, SVR model and the corresponding Q function.
## Testing a batch of SVR models
To produce the results in Table 1, after a set of SVR models have been computed, one runs:
python test_robust_svr.py --directoryname robustness_surrogates
The script will run a set of simulations for each SVR model in the folder, and print the results on the screen. For each model, we first run a continuous communication baseline (row 1 on Table 1 of paper). This is done for each model to prevent the case where models have different Q functions.

## Example
To learn the Q function for a 2 player 5 × 5 environment with default parameters, learn an SVR approximation of the robustness surrogate and test this model we can run the following lines.
```
python train_Q.py --episodes 100000
python compute_robust_svr.py --preQ Qs/trained_Q_5x5.pickle  --alpha 0.2 --nu 0.01 --C 100 --gamma 1
python test_robust_svr.py --directoryname robustness_surrogates
```
##Reproducing the results in Table 1
To reproduce the exact results, we provide the trained SVR models and Q functions used in the paper experiments. Download the zip file from (email authors for password): https://surfdrive.surf.nl/files/index.php/s/1r4yE3Bh9ZH9cgj

Copy the pickle SVR models into the folder robustness surrogates and run:
```
python test_robust_svr.py --directoryname robustness_surrogates
```

This will begin printing on the screen the results for the simulations with the different SVR models. These are the results presented on Table 1.

## Hyperparameters of SVR

Each value of $\alpha$ in Table 1 in the paper entails the solution of a scenario approach optimization using a random sample of points 
$X_{S}=\{(x,\Gamma_{\alpha}(x))\}$ of size $S=10^{5}$.
For each SVR model the hyper-parameters were tuned to obtain comparable results between them, considering that the target values $\Gamma_{\alpha}$ produced quite different results depending on the considered 
$\alpha$. In the table below we collect some more details on the computed SVR models used to approximate the function 
$\Gamma_{\alpha}$. All the SVR models were computed using scikit-learn's 
$\mu$-SVR implementation.

| $\alpha$ | $\rho$ | $\tau$ | $\kappa^*$ | $R^2$ | $\frac{s^*}{S}$ |
| ---      | ---      | ---      | ---      | ---      | ---      |
| $0.4$ | $0.01$ | $100$ | $0.00067$ | $0.999$ | $0.067$|
| $0.5$ | $0.01$ | $100$ | $0.0510$ | $0.907$ | $0.132$|
| $0.6$ | $0.1$ | $100$ | $0.0350$ | $0.995$ | $0.187$|
| $0.7$ | $0.1$ | $100$ | $0.0340$ | $0.998$ | $0.103$|
| $0.8$ | $0.1$ | $100$ | $0.0131$ | $0.998$ | $0.063$|
| $0.9$ | $0.1$ | $100$ | $0.01485$ | $0.999$ | $0.084$|

Please note that the scikit-learn implementation uses different nomenclature for the parameters compared to the paper. The equivalence is:
- $\rho=\mu$
- $\tau=C$
- $\kappa=\epsilon$

All the SVR models were computed with a radial basis function kernel due to the fact that, intuitively, similar state vectors would yield similar robustness values. The obtained prediction scores 
$R^2$ are relatively high, which could indicate some over-fitting in the models. However, this does not seem to be a problem in the implementation since 
1. The sample of training points is large enough compared to the complexity of the function $\Gamma_\alpha$
2. The approximations are very conservative. 
