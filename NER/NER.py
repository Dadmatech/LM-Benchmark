import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# print(torch.cuda.is_available())

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
import os



def read_data(path):
    sentences = []
    ner_tags = []
    with open(path, "r",  encoding='utf-8') as file:
        sentence = []
        ner_tag = []
        flag = 0
        for line in file:
            line = line.strip()
            # print(line.strip())
            if line != "":
                word, tag = line.split("\t")
                sentence.append(word)
                ner_tag.append(tag)
                flag = 0
            elif line=="" and flag==0:
                flag = 1
            elif line =="" and flag==1:
                sentences.append(sentence)
                ner_tags.append(ner_tag)
                sentence = []
                ner_tag = []
                flag = 2
            elif flag==2:
                flag = 0

    return sentences, ner_tags

current_directory = os.path.dirname(__file__)
parent_directory = os.path.dirname(current_directory)
# grandparent_directory = os.path.dirname(parent_directory)
models_directory = os.path.join(parent_directory, 'base_models')

train_sentences, train_ner_tags = read_data(parent_directory + '/datasets/ner_data/train.conll')
test_sentences, test_ner_tags = read_data(parent_directory + '/datasets/ner_data/test.conll')

print(train_sentences[1])
print(train_ner_tags[1])
print("******************************************************************************************************")

def get_counts(sentences, ner_tags):
    # Total counts
    total_words = sum(len(sentence) for sentence in sentences)
    total_tags = sum(len(tag_list) for tag_list in ner_tags)

    # Unique counts
    unique_words = set(word for sentence in sentences for word in sentence)
    unique_tags = set(tag for tag_list in ner_tags for tag in tag_list)

    return total_words, total_tags, len(unique_words), len(unique_tags)


# Get counts for train data
train_total_words, train_total_tags, train_unique_words, train_unique_tags = get_counts(train_sentences, train_ner_tags)
print(f"Train data - Total words: {train_total_words}, Total tags: {train_total_tags}, Unique words: {train_unique_words}, Unique tags: {train_unique_tags}")

# Get counts for test data
test_total_words, test_total_tags, test_unique_words, test_unique_tags = get_counts(test_sentences, test_ner_tags)
print(f"Test data - Total words: {test_total_words}, Total tags: {test_total_tags}, Unique words: {test_unique_words}, Unique tags: {test_unique_tags}")

print("******************************************************************************************************")


from collections import Counter

def get_tag_counts(ner_tags):
    # Flatten the tag lists and compute counts
    flat_tags = [tag for tag_list in ner_tags for tag in tag_list]
    tag_counts = Counter(flat_tags)
    return tag_counts

# Using the function on your train and test data
train_tag_counts = get_tag_counts(train_ner_tags)
test_tag_counts = get_tag_counts(test_ner_tags)

# Printing the counts for train data
print("Train data tag counts:")
for tag, count in train_tag_counts.items():
    print(f"{tag}: {count}")

print("\nTest data tag counts:")
for tag, count in test_tag_counts.items():
    print(f"{tag}: {count}")

print("******************************************************************************************************")

# Create label2id and id2label using the keys (tags) from train_tag_counts
label2id = {tag: idx for idx, tag in enumerate(train_tag_counts.keys())}
id2label = {idx: tag for tag, idx in label2id.items()}

print("label2id:", label2id)
print(id2label)

print("******************************************************************************************************")
def data_to_dataframe(sentences, ner_tags):
    """Convert lists of words and NER tags into a DataFrame."""
    # Convert lists of words and tags to single strings
    sentence_strs = [' '.join(sentence) for sentence in sentences]
    ner_tag_strs = [','.join(tags) for tags in ner_tags]

    # Create and return a DataFrame
    return pd.DataFrame({
        'sentence': sentence_strs,
        'word_labels': ner_tag_strs
    })


train_df = data_to_dataframe(train_sentences, train_ner_tags)
test_df = data_to_dataframe(test_sentences, test_ner_tags)

train_df.head()

print("******************************************************************************************************")
print(test_df.head())
print("******************************************************************************************************")
print(train_df.iloc[41].sentence)
print("******************************************************************************************************")
print(train_df.iloc[41].word_labels)
print("******************************************************************************************************")



#### **Preparing the dataset and dataloader**
from transformers import AutoTokenizer, AutoModelForMaskedLM
MAX_LEN = 64
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10
tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = []
    labels = []

    sentence = sentence.strip()

    for word, label in zip(sentence.split(), text_labels.split(",")):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        # step 1: tokenize (and adapt corresponding labels)
        sentence = self.data.sentence[index]
        word_labels = self.data.word_labels[index]
        tokenized_sentence, labels = tokenize_and_preserve_labels(sentence, word_labels, self.tokenizer)

        # step 2: add special tokens (and corresponding labels)
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"] # add special tokens
        labels.insert(0, "O") # add outside label for [CLS] token
        labels.insert(-1, "O") # add outside label for [SEP] token

        # step 3: truncating/padding
        maxlen = self.max_len

        if (len(tokenized_sentence) > maxlen):
          # truncate
          tokenized_sentence = tokenized_sentence[:maxlen]
          labels = labels[:maxlen]
        else:
          # pad
          tokenized_sentence = tokenized_sentence + ['[PAD]'for _ in range(maxlen - len(tokenized_sentence))]
          labels = labels + ["O" for _ in range(maxlen - len(labels))]

        # step 4: obtain the attention mask
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]

        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)

        label_ids = [label2id[label] for label in labels]
        # the following line is deprecated
        #label_ids = [label if label != 0 else -100 for label in label_ids]

        return {
              'ids': torch.tensor(ids, dtype=torch.long),
              'mask': torch.tensor(attn_mask, dtype=torch.long),
              #'token_type_ids': torch.tensor(token_ids, dtype=torch.long),
              'targets': torch.tensor(label_ids, dtype=torch.long)
        }

    def __len__(self):
        return self.len

print("TRAIN Dataset: {}".format(train_df.shape))
print("TEST Dataset: {}".format(test_df.shape))

training_set = dataset(train_df, tokenizer, MAX_LEN)
testing_set = dataset(test_df, tokenizer, MAX_LEN)

print("******************************************************************************************************")
print(training_set[1])
print("******************************************************************************************************")
training_set[1]["ids"]
print("******************************************************************************************************")
# print the first 30 tokens and corresponding labels
for token, label in zip(tokenizer.convert_ids_to_tokens(training_set[0]["ids"][:30]), training_set[0]["targets"][:30]):
  print('{0:10}  {1}'.format(token, id2label[label.item()]))
print("******************************************************************************************************")
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)
print("******************************************************************************************************")


#### **Defining the model**

model = BertForTokenClassification.from_pretrained("albert-base-v2",
                                                   num_labels=len(id2label),
                                                   id2label=id2label,
                                                   label2id=label2id)
model.to(device)



ids = training_set[0]["ids"].unsqueeze(0)
mask = training_set[0]["mask"].unsqueeze(0)
targets = training_set[0]["targets"].unsqueeze(0)
ids = ids.to(device)
mask = mask.to(device)
targets = targets.to(device)
outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
initial_loss = outputs[0]
print(initial_loss)

print("******************************************************************************************************")
tr_logits = outputs[1]
print(tr_logits.shape)
print("******************************************************************************************************")

# Freeze the base model and fine-tune the BertForTokenClassification

# Create a list of parameters to be optimized (only those of the BertForTokenClassification)
optimizer_parameters = [
    {'params': model.classifier.parameters(), 'lr': LEARNING_RATE}
]
# Set requires_grad to False for all parameters of the base model
for param in model.base_model.parameters():
    param.requires_grad = False
# Create an optimizer that only optimizes the parameters of the BertForTokenClassification
optimizer = torch.optim.Adam(params=optimizer_parameters, lr=LEARNING_RATE)

# optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

from tqdm import tqdm

# Defining the training function on the 80% of the dataset for tuning the bert model
def train(epoch):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()

    progress_bar = tqdm(enumerate(training_loader), total=len(training_loader), ncols=80, desc=f"Epoch {epoch + 1}")

    for idx, batch in progress_bar:

        ids = batch['ids'].to(device, dtype = torch.long)
        mask = batch['mask'].to(device, dtype = torch.long)
        targets = batch['targets'].to(device, dtype = torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
        loss, tr_logits = outputs.loss, outputs.logits
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if idx % 100==0:
            loss_step = tr_loss/nb_tr_steps
            print(f"Training loss per 100 training steps: {loss_step}")

        # compute training accuracy
        flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
        active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
        targets = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        tr_preds.extend(predictions)
        tr_labels.extend(targets)

        tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=MAX_GRAD_NORM
        )

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()
        # After you're done using data
        del batch, ids, mask, targets
    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")
print("******************************************************************************************************")


###  *** **train the model** ***


for epoch in range(EPOCHS):
    print(f"Training epoch: {epoch + 1}")
    train(epoch)

#### **Evaluating the model**

def valid(model, testing_loader):
    # put model in evaluation mode
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):

            ids = batch['ids'].to(device, dtype = torch.long)
            mask = batch['mask'].to(device, dtype = torch.long)
            targets = batch['targets'].to(device, dtype = torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, eval_logits = outputs.loss, outputs.logits

            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += targets.size(0)

            if idx % 100==0:
                loss_step = eval_loss/nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")

            # compute evaluation accuracy
            flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
            active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            eval_labels.extend(targets)
            eval_preds.extend(predictions)

            tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy
            # After you're done using data
            del batch, ids, mask, targets
        torch.cuda.empty_cache()

    #print(eval_labels)
    #print(eval_preds)

    labels = [id2label[id.item()] for id in eval_labels]
    predictions = [id2label[id.item()] for id in eval_preds]

    #print(labels)
    #print(predictions)

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    return labels, predictions

labels, predictions = valid(model, testing_loader)

print("******************************************************************************************************")
from seqeval.metrics import classification_report

print(classification_report([labels], [predictions]))
print("******************************************************************************************************")


#### **Inference**


sentence = "ابراهیم رییسی به همدان رفت."

inputs = tokenizer(sentence, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")

# move to gpu
ids = inputs["input_ids"].to(device)
mask = inputs["attention_mask"].to(device)
# forward pass
outputs = model(ids, mask)
logits = outputs[0]

active_logits = logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
token_predictions = [id2label[i] for i in flattened_predictions.cpu().numpy()]
wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

word_level_predictions = []
for pair in wp_preds:
  if (pair[0].startswith(" ##")) or (pair[0] in ['[CLS]', '[SEP]', '[PAD]']):
    # skip prediction
    continue
  else:
    word_level_predictions.append(pair[1])

# we join tokens, if they are not special ones
str_rep = " ".join([t[0] for t in wp_preds if t[0] not in ['[CLS]', '[SEP]', '[PAD]']]).replace(" ##", "")
print(str_rep)
print(word_level_predictions)

print("******************************************************************************************************")
from transformers import pipeline

pipe = pipeline(task="token-classification", model=model.to("cpu"), tokenizer=tokenizer, aggregation_strategy="simple")
pipe("سلام اسم من علی است و در ایران زندگی میکنم.")
print("******************************************************************************************************")


#### **Saving the model for future use**

tokenizer.save_pretrained("./fine_tuned_model_alberta/")
model.save_pretrained("./fine_tuned_model_alberta/")
