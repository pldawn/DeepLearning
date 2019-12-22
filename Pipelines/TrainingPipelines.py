from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Pipelines import Pipelines


class TrainingPipelines(Pipelines):
    def __init__(self):
        super(TrainingPipelines, self).__init__()
        self.allowed_process = ['training_datasets', 'eval_datasets', 'preprocessings', 'models', 'optimizers',
                                'learning_rates', 'losses', 'training_steps', 'eval_steps', 'postprocessings',
                                "metrics"]
        self.required_process = ['training_datasets', 'models', 'optimizers', 'losses', 'training_steps', "metrics"]
