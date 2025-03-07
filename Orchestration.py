import nbformat
from nbconvert import PythonExporter
from nbconvert.preprocessors import ExecutePreprocessor

notebooks = [f'EDFA{i}_Model.ipynb' for i in range(1, 7)]

for notebook in notebooks:
    with open(notebook, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, nbformat.NO_CONVERT)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    nb_out = ep.preprocess(nb)