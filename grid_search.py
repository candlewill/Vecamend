__author__ = 'NLP'
from sklearn.grid_search import ParameterGrid
import numpy as np
from save_data import dump_picle


def my_function(param1, param2):
    return param1+param2

if __name__=='__main__':
    param_grid = {'a': [1, 2], 'b': [4, 2]}
    param_fitness = []

    grid = ParameterGrid(param_grid)

    for params in grid:
        print('calculating... parameter: %s' % str(params))
        score = my_function(params['a'], params['b'])
        print('Score: %s' % score)
        param_fitness.append(score)

    print('grid search complete.')
    # return the best fitness value and its settings
    best_fitness = np.min(np.array(param_fitness))
    best_ind = np.where(np.array(param_fitness)==best_fitness)[0]
    print('best fitness: %s' % best_fitness)
    print('best setting: %s' % str(list(grid)[best_ind]))

    dump_picle((param_grid, param_fitness), './tmp/grid_search_result.p')