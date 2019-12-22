from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Schedule:
    def __init__(self, **kwargs):
        self.training_datasets = []
        self.eval_datasets = []
        self.preprocessings = []
        self.models = []
        self.optimizers = []
        self.learning_rates = []
        self.losses = []
        self.training_steps = []
        self.eval_steps = []
        self.predict_steps = []
        self.metrics = []
        self.postprocessings = []
        self.pipelines = None
        self.results = []
        self.kwargs = kwargs

    def add_training_datasets(self, training_datasets):
        self.training_datasets = training_datasets

    def add_eval_datasets(self, eval_datasets):
        self.eval_datasets = eval_datasets

    def add_preprocessings(self, preprocessings):
        self.preprocessings = preprocessings

    def add_models(self, models):
        self.models = models

    def add_optimizers(self, optimizers):
        self.optimizers = optimizers

    def add_learning_rates(self, learning_rates):
        self.learning_rates = learning_rates

    def add_losses(self, losses):
        self.losses = losses

    def add_training_steps(self, training_steps):
        self.training_steps = training_steps

    def add_eval_steps(self, eval_steps):
        self.eval_steps = eval_steps

    def add_predict_steps(self, predict_steps):
        self.predict_steps = predict_steps

    def add_metrics(self, metrics):
        self.metrics = metrics

    def add_postprocessings(self, postprocessings):
        self.postprocessings = postprocessings

    def add_pipelines(self, pipelines):
        self.pipelines = pipelines

    def run(self):
        if self.pipelines is None:
            self.get_default_pipelines()

        self.run_pipelines(self.pipelines)

        return self.results

    def get_default_pipelines_core(self):
        for process in self.__dict__:
            if process in self.pipelines.allowed_process:
                if len(self.__dict__[process]) > 0:
                    self.pipelines.add_process(process=process, method_indices='all')
                    self.pipelines.add_process_method_num(process=process, num=len(self.__dict__[process]))

        self.pipelines.build()

    def get_default_pipelines(self):
        pass

    def run_pipelines(self, pipelines):
        pass
