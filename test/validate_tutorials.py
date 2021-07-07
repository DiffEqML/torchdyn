import papermill as pm

# TODO: extend tutorial automated valid
TORCHDYN_NOTEBOOKS = [
    '00_quickstart.ipynb']
    #,
    #'module1-neuralde/01_neural_ode_cookbook.ipynb',
    #'module2-numerics/02_hypersolver_odeint.ipynb']


for notebook in TORCHDYN_NOTEBOOKS:
    path_to_notebook = f'tutorials/{notebook}'
    pm.execute_notebook(path_to_notebook, path_to_notebook)
