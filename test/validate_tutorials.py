import papermill as pm

# TODO: extend tutorial automated valid
TORCHDYN_NOTEBOOKS = ['00_quickstart.ipynb']#
                     #, '01_neural_ode_cookbook.ipynb', '03_crossing_trajectories.ipynb',
                     # '04_augmentation_strategies.ipynb', '05_generalized_adjoint.ipynb', '06_higher_order.ipynb', '07a_continuous_normalizing_flows.ipynb',
                     # '07b_ffjord.ipynb', '08_hamiltonian_nets.ipynb', '09_lagrangian_nets.ipynb', '10_stable_neural_odes.ipynb', '11_gde_node_classification.ipynb']

for notebook in TORCHDYN_NOTEBOOKS:
    path_to_notebook = f'tutorials/{notebook}'
    pm.execute_notebook(path_to_notebook, path_to_notebook)
