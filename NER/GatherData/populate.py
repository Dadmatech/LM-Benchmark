import csv

from ast import literal_eval
from glob import glob
from hashlib import md5 as sample_id

class NerDatasetNormalizer(object):
  def __init__(self):
    self.tag_map = {}

  def normalize_tag(self, tag):
    if tag in self.tag_map:
      return self.tag_map[tag]
    return "O"

class ArmanNerDatasetNormalizer(NerDatasetNormalizer):
  def __init__(self, csv_file_path):
    super().__init__()
    self.csv_file_path = csv_file_path
    self.csv_file = None
    self.tag_map = {
      "B-pers": "B-PER",
      "I-pers": "I-PER",
      "B-org": "B-ORG",
      "I-org": "I-ORG",
      "B-loc": "B-LOC",
      "I-loc": "I-LOC"
    }
  
  def __enter__(self):
    self.csv_file = open(self.csv_file_path)
    self.reader = csv.reader(self.csv_file)
    return self.__sample_generator()

  def __exit__(self, *exception_info):
    self.csv_file.close()
    self.csv_file = None

  def __sample_generator(self):
    if self.csv_file is None:
      raise EnvironmentError("Called outside a width statement :(")
    
    # Skip the title row
    for _ in self.reader:
      break
    
    for words, tags in self.reader:
      yield literal_eval(words), [self.normalize_tag(t) for t in literal_eval(tags)]

class ConllLikeDatasetNormalizer(NerDatasetNormalizer):
  def __init__(self, words_file_path):
    super().__init__()
    self.words_file_path = words_file_path
    self.words_file = None
    self.tag_map = {
      "B-PER": "B-PER",
      "I-PER": "I-PER",
      "B-ORG": "B-ORG",
      "I-ORG": "I-ORG",
      "B-LOC": "B-LOC",
      "I-LOC": "I-LOC",
    }
  
  def __enter__(self):
    self.words_file = open(self.words_file_path)
    return self.__sample_generator()

  def __exit__(self, *exception_info):
    self.words_file.close()
    self.words_file = None

  def __sample_generator(self):
    if self.words_file is None:
      raise EnvironmentError("Called outside a width statement :(")
    
    data = [l.strip().split("\t") if l.strip() else None for l in self.words_file.readlines()]
    
    i = 0
    try:
      j = data.index(None)
    except ValueError:
      j = len(data)
    while j < len(data):
      if i < j:
        yield [e[0] for e in data[i:j]], [self.normalize_tag(e[-1]) for e in data[i:j]]
      i = j + 1
      try:
        j = data.index(None, i)
      except ValueError:
        j = len(data)
    if i < len(data):
      yield [e[0] for e in data[i:]], [self.normalize_tag(e[-1]) for e in data[i:]]

class NerDataCollection(object):
  def __init__(self, f):
    self.unique_samples_count = 0
    self.repetitive_samples_count = 0
    self.token_count = 0
    self.f = f
    self.data = set()

  def append(self, words, tags):
    key = sample_id(" ".join(words).encode()).digest()
    if key in self.data:
      self.repetitive_samples_count += 1
      return
    self.unique_samples_count += 1
    self.token_count += len(words)
    self.data.add(key)
    for w, t in zip(words, tags):
      print(f"{w}\t{t}\n", file=self.f)
    print("\n", file=self.f)

  def print_info(self):
    print("# of unique samples: ", self.unique_samples_count)
    print("# of repititions: ", self.repetitive_samples_count)
    print("# of tokens: ", self.token_count)

  def data(self):
    return self.data.values()

datasets = [
  ("arman-ner/arman-ner_train.csv", "arman-ner/arman-ner_test.csv", ArmanNerDatasetNormalizer),
  ("ParsNER/persian/train/*.pos", "ParsNER/persian/test/*.pos", ConllLikeDatasetNormalizer),
  ("Persian-NER/Persian-NER-part*.txt", None, ConllLikeDatasetNormalizer)
]

with open("./train.conll", "w") as f0, open("./test.conll", "w") as f1:
  train_data = NerDataCollection(f0)
  test_data = NerDataCollection(f1)
  for train_glob_pattern, test_glob_pattern, normalizer_class in datasets:
    for file_path in glob(train_glob_pattern):
      print(f"Normalizing train data from `{file_path}`")
      with normalizer_class(file_path) as normalized_items:
        for words, tags in normalized_items:
          train_data.append(words, tags)
    if test_glob_pattern:
      for file_path in glob(test_glob_pattern):
        print(f"Normalizing test data from `{file_path}`")
        with normalizer_class(file_path) as normalized_items:
          for words, tags in normalized_items:
            test_data.append(words, tags)

    print("Train data information until now ...")
    print("====================================")
    train_data.print_info()

    print("Test data information until now:")
    print("================================")
    test_data.print_info()    

print("Train data information:")
print("======================")
train_data.print_info()

print("Test data information:")
print("======================")
test_data.print_info()
