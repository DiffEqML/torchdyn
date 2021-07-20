import glob 
import os
import papermill as pm

path_cwd = os.getcwd()
os.chdir('..')
path = os.getcwd()

TORCHDYN_TUTORIALS_MODULES = [
    "module1-neuralde",
    "module2-numerics"
]
for i in range(len(TORCHDYN_TUTORIALS_MODULES)):
    tutorial_paths = glob.glob(path+"/tutorials/"+TORCHDYN_TUTORIALS_MODULES[i]+"/*.ipynb")
    for j, path_to_notebook in enumerate(tutorial_paths):
        path_to_output = path + f"/tutorials/local_nbrun_m{i}_{j}"
        parameters=dict(dry_run=True)
        pm.execute_notebook(path_to_notebook, path_to_output, parameters=parameters)

# import papermill as pm


# TORCHDYN_NOTEBOOKS_PATHS = [
#     '00_quickstart.ipynb',
#     'module1-neuralde/m1a_neural_ode_cookbook.ipynb',
#     'module1-neuralde/m1b_crossing_trajectories.ipynb',
#     'module1-neuralde/m1c_augmentation_strategies.ipynb',
#     'module1-neuralde/m1d_higher_order.ipynb',
#     'module1-neuralde/m1e_crossing_trajectories',
#     'module1-neuralde/m1f_augmentation_strategies',
#     'module1-neuralde/m1g_higher_order',
#     'module2-numerics/01_hypersolver_odeint.ipynb',
#     'module2-numerics/02_multiple_shooting.ipynb',
#     'module2-numerics/03_hybrid_odeint.ipynb',
#     'module2-numerics/04_generalized_adjoint.ipynb']


# for path in TORCHDYN_NOTEBOOKS_PATHS:
#     notebook_path = path.split('/')
#     if len(notebook_path) == 1: 
#         notebook = notebook_path[0]
#         path_to_notebook = f'tutorials/{notebook}'
#     else: 
#         module, notebook = notebook_path
#         path_to_notebook = f'tutorials/{module}/{notebook}'
#     path_to_output = f'tutorials/local_nbrun_{notebook}'
#     parameters=dict(dry_run=True)
#     pm.execute_notebook(path_to_notebook, path_to_output, parameters=parameters)
