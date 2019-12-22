from Schedule import TrainingSchedule
import pprint


ts = TrainingSchedule()


def fn(**kwargs):
    return fn


ts.add_training_datasets([fn, fn])
ts.add_eval_datasets([fn])
ts.add_preprocessings([fn])
ts.add_models([fn, fn])
ts.add_optimizers([fn])
ts.add_learning_rates([fn, fn])
ts.add_losses([fn, fn])
ts.add_training_steps([fn])
ts.add_eval_steps([fn])
ts.add_metrics([fn])
ts.add_postprocessings([fn])

ts.run()
pprint.pprint(len(ts.results))
pprint.pprint(ts.results)
