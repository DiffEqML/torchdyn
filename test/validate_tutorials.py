import papermill as pm

# TODO: extend tutorial automated validation, dry run parameter for quick training
TORCHDYN_NOTEBOOKS_PATHS = [
    '00_quickstart.ipynb',
    'module1-neuralde/01_neural_ode_cookbook.ipynb']
    #'module2-numerics/02_hypersolver_odeint.ipynb']


for path in TORCHDYN_NOTEBOOKS_PATHS:
    notebook_path = path.split('/')
    print(notebook_path)
    if len(notebook_path) == 1: 
        notebook = notebook_path[0]
        path_to_notebook = f'tutorials/{notebook}'
    else: 
        module, notebook = notebook_path
        path_to_notebook = f'tutorials/{module}/{notebook}'
    path_to_output = f'tutorials/local_nbrun_{notebook}'
    parameters=dict(dry_run=True)
    pm.execute_notebook(path_to_notebook, path_to_output, parameters=parameters)
