from nni.algorithms.hpo.gridsearch_tuner import GridSearchTuner
from nni.common.hpo_utils import ParameterSpec, deformat_parameters, format_search_space

class MarlTuner(GridSearchTuner):
    
    def update_search_space(self, space):
        self.space = format_search_space(space)
        if not self.space:  # the tuner will crash in this case, report it explicitly
            raise ValueError('Search space is empty')
        print("self.space", self.space)
        self._init_grid()