import math
from copy import deepcopy as dcp
from itertools import chain

from ato.adict import ADict
from ato.hyperopt.base import HyperOpt, DistributedMixIn, GridSpaceMixIn


class HyperBand(HyperOpt, GridSpaceMixIn):
    def __init__(self, scope, search_spaces, halving_rate, num_min_samples, tracker=None, mode='max'):
        if halving_rate <= 0 or halving_rate >= 1:
            raise ValueError(f'halving_rate must be greater than 0.0 but less than 1.0, but got {halving_rate}.')
        if num_min_samples < 1:
            raise ValueError(f'num_min_samples must be greater than or equal to 1, but got {num_min_samples}.')
        super().__init__(scope, search_spaces, tracker, mode)
        self.halving_rate = halving_rate
        self.num_min_samples = num_min_samples
        self.distributions = self.prepare_distributions(self.config, self.search_spaces)

    @classmethod
    def prepare_distributions(cls, base_config, search_spaces):
        distributions = super().prepare_distributions(base_config, search_spaces)
        for distribution in distributions:
            distribution.__num_halved__ = 0
        return distributions

    def main(self, func):
        def launch(*args, **kwargs):
            logs = []
            distributions = self.distributions
            while len(distributions) >= self.num_min_samples:
                results = self.estimate(func, distributions, *args, **kwargs)
                results.sort(key=lambda item: item.__metric__, reverse=self.mode == 'max')
                logs.append(results)
                distributions = []
                for config in results[:int(len(results)*self.halving_rate)]:
                    config.__num_halved__ += 1
                    distributions.append(config)
            last_config = logs[-1][0]
            metric = self.estimate_single_run(func, last_config, *args, **kwargs)
            best_config = dcp(last_config)
            best_config.__metric__ = metric
            logs.append([best_config])
            return ADict(config=best_config, metric=metric, logs=logs)
        return launch

    def estimate(self, estimator, distributions, *args, **kwargs):
        results = []
        for config in distributions:
            config = dcp(config)
            self.scope.config = config
            metric = self.estimate_single_run(estimator, config, *args, **kwargs)
            config.__metric__ = metric
            results.append(config)
        return results

    def estimate_single_run(self, estimator, config, *args, **kwargs):
        self.scope.config = config
        return self.scope(estimator)(*args, **kwargs)

    def num_generations(self):
        max_size = len(self.distributions)
        min_size = self.num_min_samples
        return math.ceil(math.log(min_size/max_size)/math.log(self.halving_rate))

    def compute_optimized_initial_training_steps(self, max_steps):
        num_generations = self.num_generations()
        min_steps = max_steps*math.pow(self.halving_rate, num_generations)
        return [
            *(math.ceil(min_steps/math.pow(self.halving_rate, index)) for index in range(1, num_generations)),
            max_steps
        ]


class DistributedHyperBand(DistributedMixIn, HyperBand, GridSpaceMixIn):
    def __init__(
        self,
        scope,
        search_spaces,
        halving_rate,
        num_min_samples,
        tracker=None,
        mode='max',
        rank=0,
        world_size=1,
        backend='pytorch'
    ):
        DistributedMixIn.__init__(self, rank, world_size, backend)
        HyperBand.__init__(self, scope, search_spaces, halving_rate, num_min_samples, tracker, mode)

    def estimate(self, estimator, distributions, *args, **kwargs):
        batch_size = math.ceil(len(distributions)/self.world_size)
        distributions = distributions[self.rank*batch_size:(self.rank+1)*batch_size]
        results = super().estimate(estimator, distributions, *args, **kwargs)
        results = list(chain(*self.all_gather_object(results)))
        return results

    def estimate_single_run(self, estimator, config, *args, **kwargs):
        result = super().estimate(estimator, config, *args, **kwargs)
        self.destroy()
        return result


