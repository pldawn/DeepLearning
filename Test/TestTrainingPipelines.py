from Pipelines import TrainingPipelines


tp1 = TrainingPipelines()
tp2 = TrainingPipelines()

tp1.add_process_method_num('training_datasets', 2)
tp1.add_process_method_num('preprocessings', 3)
tp1.add_process_method_num('models', 3)
tp1.add_process_method_num('optimizers', 4)
tp1.add_process_method_num('losses', 1)
tp1.add_process_method_num('training_steps', 1)
tp1.add_process_method_num('postprocessings', 1)

tp1.add_process('training_datasets', 'all')
tp1.add_process('preprocessings', [0,2])
tp1.add_process('models', 'all')
tp1.add_process('optimizers', (1,3))
tp1.add_process('losses', 0)
tp1.add_process('training_steps', 'all')
tp1.add_process('postprocessings', 'all')

tp2.add_sub_pipelines(tp1)

tp1.build()

n = 0
for i in tp1.pipelines_iterator:
    print(type(i), dict(i))
    n += 1
print(n)
