# Reparameterized Variational Rejection Sampling

This repository contains an example script [logistic_regression.py](logistic_regression.py) that reproduces the main 
results for the logistic regression experiment described in Section 7.2 and Figure 4 of
[Reparameterized Variational Rejection Sampling](https://arxiv.org/abs/2309.14612) (Martin Jankowiak & Du Phan,
AISTATS 2024). We include a single dataset ([bank.csv](bank.csv)). The main RVRS functionality, 
which is implemented as a [NumPyro](num.pyro.ai) AutoGuide (i.e. variational distribution), can be found in [rvrs.py](rvrs.py).

To run the script we suggest the following workflow:

- create a new conda environment:     `conda create -n rvrs python=3.9`
- activate the new environment:       `conda activate rvrs`
- install requirements:               `pip install -r requirements.txt`
- examine script arguments:           `python logistic_regression.py --help`
- run script with default arguments:  `python logistic_regression.py`
