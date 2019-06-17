# Optimal Continuous DR-Submodular Maximization and Applications to Provable Mean Field Inference

This repository collects source code for the paper:

"Optimal Continuous DR-Submodular Maximization and Applications to Provable Mean Field Inference"

ICML 2019. Yatao A. Bian, Joachim M. Buhmann and Andreas Krause

### File Structure:

 - functions/:  contains utility files
    - exp_specs.py: contain specifications for the experiments
    - utils.py: utility functions
    - process_results/: contains notebook to calculate experimental stats
 - main.py: the main file to run different experiments
 - data/: contains FLID models

## Usage:

The absl flags define the main usage of the code. Specifically,

`flags.DEFINE_integer('problem_id', 1, 'Options: 1: ELBO, 2: PA-ELBO.')`
- For ELBO, we run different algorithms on the ELBO objective
- For PA-ELBO, we run different algorithms on the PA-ELBO objective


`flags.DEFINE_string('mode', 'run', 'Options: run: run algorithms; stats: get experimental statistics.')`

- 'run': run different algorithms/solvers, dump the results into pickle file and plot figures
- 'stats': generate function values returned in all the experiments and dump them into a pickle file.

`flags.DEFINE_boolean('debug', True, 'Whether it is in debug mode.')`
- In debug mode, one only run solvers on one fold of FLID model, which is much faster than the non-debug mode.
- Be careful, it may takes hours to run the non-debug mode, especially for the PA-ELBO objectives.

## Example:

Let us consider an example for the ELBO objective.

Firstly navigate to the folder in your
command line and run:

`$ python main.py  --mode=run --debug=False --problem_id=1`

Results will be stored in the ./results folder.
To get the experimental stats file, you can then run:

`$ python main.py  --mode=stats --debug=False --problem_id=1`

A pickle file called `optf_1epoch.pkl` will be generated in the same result folder. In order to see more details of experimental stats, you can go to the folder `process_results/` and play with the notebook `process_results.ipynb`.


## Dependencies:

 - The code has been tested on Ubuntu 17.10, 64 bits with Python 3.6. It should work with other OS with little change.

## Copyright:

 Copyright (2019) [Yatao (An) Bian <yatao.bian@gmail.com> | yataobian.com].  
 Please cite the above paper if you use this code in your work.
