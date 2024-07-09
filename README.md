# nonlocality_certification

The file ``codes.py`` contains all the codes that are implemented in the  ``nonlocality_certification`` jupyter-notebook. The main routine is ``optimal_value(...)``, 
which optimizes the following function


$$
R=\max_{s}\frac{\mathcal{Q}(s)-\Delta \mathcal{Q}(s)+dm}{\mathcal{C}(s)+dm},
$$

with $\mathcal{Q}(s)$ </a> and $\mathcal{C}(s)$ </a> the Quantum and the Local Hidden values, respectively, of a Bell inequality and $\Delta \mathcal{Q}(s)$ </a> is the experimental error of the Quantum value. This optimization requires the data stored in the ``experimental_data`` folder. 

#### Gómez, S., Uzcátegui, D., Machuca, I. et al. Optimal strategy to certify quantum nonlocality. [Sci Rep 11, 20489 (2021).](https://doi.org/10.1038/s41598-021-99844-2)
