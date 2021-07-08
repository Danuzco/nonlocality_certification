# nonlocality_certification

The file ``codes.py`` contains all the codes that are implemented in the  ``nonlocality_certification`` jupyter-notebook. The main routine is ``optimal_value(...)``, 
which optimizes the following function

<a href="https://www.codecogs.com/eqnedit.php?latex=R=\max_{s}\dfrac{\mathcal{Q}(s)-\Delta&space;\mathcal{Q}(s)&plus;(dm)^2}{\mathcal{C}(s)&plus;(dm)^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R=\max_{s}\dfrac{\mathcal{Q}(s)-\Delta&space;\mathcal{Q}(s)&plus;(dm)^2}{\mathcal{C}(s)&plus;(dm)^2}" title="R=\max_{s}\dfrac{\mathcal{Q}(s)-\Delta \mathcal{Q}(s)+(dm)^2}{\mathcal{C}(s)+(dm)^2}" /></a>

with <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathcal{Q}(s)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\mathcal{Q}(s)" title="\mathcal{Q}(s)" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathcal{C}(s)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\mathcal{C}(s)" title="\mathcal{C}(s)" /></a> the Quantum and Classical values, respectively, of a Bell inequality. <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\Delta\mathcal{Q}(s)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\Delta\mathcal{Q}(s)" title="\Delta\mathcal{Q}(s)" /></a> is the experimental error of the Quantum value. This optimization requires the data stored in the ``experimental_data`` folder. 
