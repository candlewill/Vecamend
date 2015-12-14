__author__ = 'NLP'
from sklearn.grid_search import ParameterGrid
def my_function(param1, param2):
    pass

if __name__=='__main__':
    param_grid = {'a': [1, 2], 'b': [True, False]}

    grid = ParameterGrid(param_grid)

    for params in grid:
        my_function(params['param1'], params['param2'])