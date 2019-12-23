from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime

from Schedule import Schedule
from Pipelines import TrainingPipelines


class TrainingSchedule(Schedule):
    def __init__(self, **kwargs):
        self.start_time = datetime.strftime(datetime.now(), format="%Y-%m-%d_%H-%M-%S")
        super(TrainingSchedule, self).__init__(**kwargs)

    def get_default_pipelines(self):
        self.pipelines = TrainingPipelines()
        self.get_default_pipelines_core()

        return self.pipelines

    def run_pipelines(self, pipelines):
        if not self.pipelines.is_build:
            self.pipelines.build()

        log_dir_mark = 0
        for pipeline in self.pipelines.pipelines_iterator:
            # convert indices pipeline to concrete training pipeline
            pipeline_dict = dict(pipeline)

            log_dir = os.path.join("log_%s" % self.start_time, "%d" % 0)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            pipeline_file = os.path.join(log_dir, "pipeline.cfg")
            with open(pipeline_file, 'w') as f:
                for k, v in pipeline_dict.items():
                    f.write(str(k) + '\t' + str(v) + '\n')

            # get concrete functions
            training_datasets_fn = self.training_datasets[pipeline_dict['training_datasets']]
            models_fn = self.models[pipeline_dict['models']]
            losses_fn = self.losses[pipeline_dict['losses']]
            optimizers_fn = self.optimizers[pipeline_dict['optimizers']]
            training_steps_fn = self.training_steps[pipeline_dict['training_steps']]
            metrics_fn = self.metrics[pipeline_dict['metrics']]

            if 'preprocessings' in pipeline_dict:
                preprocessings_fn = self.preprocessings[pipeline_dict['preprocessings']]
            else:
                preprocessings_fn = None

            if 'eval_datasets' in pipeline_dict:
                eval_datasets_fn = self.eval_datasets[pipeline_dict['eval_datasets']]
            else:
                eval_datasets_fn = None

            if 'eval_steps' in pipeline_dict:
                eval_steps_fn = self.eval_steps[pipeline_dict['eval_steps']]
            else:
                eval_steps_fn = None

            if 'learning_rates' in pipeline_dict:
                learning_rates_fn = self.learning_rates[pipeline_dict['learning_rates']]
            else:
                learning_rates_fn = None

            if 'postpreprocessings' in pipeline_dict:
                postpreprocessings_fn = self.postprocessings[pipeline_dict['postpreprocessings']]
            else:
                postpreprocessings_fn = None

            # train
            if preprocessings_fn is not None:
                training_datasets_fn = preprocessings_fn(datasets_fn=training_datasets_fn)
                eval_datasets_fn = preprocessings_fn(datasets_fn=eval_datasets_fn)

            result = training_steps_fn(training_datasets_fn=training_datasets_fn,
                                       models_fn=models_fn,
                                       losses_fn=losses_fn,
                                       optimizers_fn=optimizers_fn,
                                       eval_datasets_fn=eval_datasets_fn,
                                       eval_steps_fn=eval_steps_fn,
                                       learning_rates_fn=learning_rates_fn,
                                       metrics_fn=metrics_fn,
                                       log_dir=log_dir)

            if postpreprocessings_fn is not None:
                result = postpreprocessings_fn(result)

            self.results.append((pipeline, result))
