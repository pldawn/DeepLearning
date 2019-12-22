from Datasets.ImdbDatasets import get_imdb_datasets_fn

config = "subwords32k"
split = None

training_datasets_fn, test_datasets_fn, encoder = get_imdb_datasets_fn(config=config, split=split)

training_datasets = training_datasets_fn()
test_datasets = test_datasets_fn()

for ex in training_datasets.take(1):
    print('training example:')
    print('y: ', ex['label'])
    print('x: ', ex['text'])

print(encoder.vocab_size)
