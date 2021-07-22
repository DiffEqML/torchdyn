import glob 
import os
import papermill as pm

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