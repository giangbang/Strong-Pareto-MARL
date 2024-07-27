# Experiments on Toy MOO problems
This part is based on the code from
"Multi-Task Learning with User Preferences: 
Gradient Descent with Controlled Ascent in Pareto Optimization"
by Mahapatra Debabrata and Rajan Vaibhav
in ICML 2020 ([link](https://github.com/dbmptr/EPOSearch/tree/master/toy_experiments))

### Implementation of problems
- `toy_biobjective.py`: two objectives from Lin et al.
- `toy_biobjective2.py`: two objectives in the main paper.
- `toy_biobjective3.py`: two objectives in the main paper.

### Implementation of solvers
The `solvers` module contains different solvers:
1. Linear Sclarization: `linscalar.py`
2. MGDA based MOO: `moo_mtl.py` and `min_norm_solvers_numpy.py`
3. Pareto MTL: 
	- cpu: `pmtl.py` and `min_norm_solvers_numpy.py`
	- gpu: `pmtl_gpu.py` and `min_norm_solvers.py`
4. EPO Search: `epo_search.py` and `epo_lp.py`
5. MGDA++
6. PCGrad

### Experiments

`pareto_visualize.py` compare the convergence of MGDA with/without Adam and MGDA++

`pareto_visualize2.py` compare the convergence of MGDA and MGDA++ with standard setup without presence of weak Pareto optimal solutions

`pareto_visualize2.py` compare how MGDA and MGDA++ converge to different points

