from Preprocessings.ImdbPreprocessings import get_imdb_preprocessings_fn
from Datasets.ImdbDatasets import get_imdb_datasets_fn

vocab_size = 32650
epochs = 1
shuffle_buffer = 10000
prefetch_buffer = 1000
num_parallel_calls = 5

imdb_preprocessings_fn = get_imdb_preprocessings_fn(vocab_size=vocab_size, num_parallel_calls=num_parallel_calls)

config = "subwords32k"
split = None
training_datasets_fn, test_datasets_fn, encoder = get_imdb_datasets_fn(config=config, split=split)

new_test_datasets_fn = imdb_preprocessings_fn(test_datasets_fn)
new_test_datasets = new_test_datasets_fn()

for ex in new_test_datasets.take(2):
    print('new_test_datasets:')
    print(ex)
