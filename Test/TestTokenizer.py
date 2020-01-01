from Preprocessings import CharTokenizer, JiebaTokenizer, BPETokenizer
from Datasets.TNewsDatasets import get_tnews_datasets_fn


train_fn, _, _ = get_tnews_datasets_fn()
data, label = train_fn()

tokenizer = BPETokenizer(pre_train=False)
ids = tokenizer.tokenize(data)
print(ids[0])
print(tokenizer.vocabulary)
