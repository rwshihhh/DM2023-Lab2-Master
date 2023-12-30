import os
import torch
import time
import json
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification,TrainingArguments, Trainer
import transformers
from sklearn.preprocessing import OneHotEncoder

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
MODEL_NAME = 'distilbert-base-uncased'
LEARNING_RATE = 2e-5
BATCH_SIZE = 256
EPOCH = 5
OUTPUT_DIR = './output/'+f"lr={str(LEARNING_RATE)}_ecpoch={str(EPOCH)}"


print(torch.cuda.is_available())
time.sleep(4)

"""Load Data"""
# merge two DataFrame according to 'tweet_id'
df1 = pd.merge(pd.read_csv('kaggle_data/data_identification.csv'), pd.read_csv('kaggle_data/emotion.csv'), on='tweet_id', how='outer')
    # how='outer': retain all rows even two df don't have the same 'tweet_id' in two dataframe.
# process and combine the JSON files
# Open a JSON file containing many JSON objects
with open('kaggle_data/tweets_DM.json', 'r') as json_file:
    data = [json.loads(line) for line in json_file]
data = [{'tweet_id': item['_source']['tweet']['tweet_id'], 
         'text': item['_source']['tweet']['text'], 
         '_score': item['_score'], 
         '_index': item['_index'],  
         'hashtags': item['_source']['tweet']['hashtags'],
         '_crawldate': item['_crawldate'], 
         '_type': item['_type']
         } for item in data]
df2 = pd.DataFrame(data)
df = pd.merge(df2, df1, on='tweet_id', how='inner')
# split train and test using 'identification'
# remove meaningless attributes
df_train = df[df['identification'] == 'train']
df_test = df[df['identification'] == 'test']
df_train = df_train.drop(columns=['identification','_crawldate','_index','_type'])
df_test = df_test.drop(columns=['emotion','identification','_crawldate','_index','_type'])

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
# hashtags: list of str -> str 
df_train['hashtags'] = df_train['hashtags'].apply(lambda x: ', '.join(map(str, x)))
df_test['hashtags'] = df_test['hashtags'].apply(lambda x: ', '.join(map(str, x)))
df_test_no_hashtags = df_test.drop(columns=['_score','hashtags'])
df_train_no_hashtags = df_train.drop(columns=['_score','hashtags','tweet_id'])

train_dataset = Dataset.from_pandas(df_train_no_hashtags)
test_dataset = Dataset.from_pandas(df_test_no_hashtags)
dataset = DatasetDict({
    'train': train_dataset
    # 'test': test_dataset
})


"""Pre-processing"""
tokenizer =transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
encoder = OneHotEncoder(handle_unknown='ignore')
X = [['anticipation'], ['sadness'], ['fear'], ['joy'], ['anger'], ['trust'],['disgust'], ['surprise']]
encoder.fit(X)

LABEL_COUNT = len(encoder.categories_[0])

def preprocess(dataslice):
    """ Input: a batch of your dataset
        Example: { 'text': [['sentence1'], ['setence2'], ...],
                   'label': ['label1', 'label2', ...] }
    """
    label_list=[]
    for emotion in dataslice['emotion']:
        label_list.append(encoder.transform([[emotion]]).toarray()[0])    
    encoding = tokenizer(dataslice['text'])
    output={
        'text': dataslice['text'],
        'emotion': dataslice['emotion'],
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'label': label_list
        }
    return output

# map the function to the whole dataset
processed_data = dataset.map(preprocess,    # your processing function
                             batched = True # Process in batches so it can be faster
                            )
print("\nCheck processed\n")
for i in range(5):
    print(f"{i+1}. {processed_data['train'][i]['emotion']} -> {processed_data['train'][i]['label']}")
print(processed_data)
processed_data['train'][0]
print("\nEND\n")

"""Training"""
data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
                                                           num_labels = LABEL_COUNT)
#split val dataset from train dataset
train_val_dataset = processed_data['train'].train_test_split(test_size=0.1, shuffle=True, seed=42)

training_args = TrainingArguments(
    # output_dir = OUTPUT_DIR,
    output_dir = './output/test',
    learning_rate = LEARNING_RATE,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size = BATCH_SIZE,
    num_train_epochs = EPOCH,
    logging_dir = ".output/test/logs",
    logging_strategy = "epoch",
    save_strategy = "epoch",
    evaluation_strategy='epoch', 
    report_to = "tensorboard"
    )
trainer = Trainer(
    # set your parameters here
    model = model,
    args = training_args,
    data_collator = data_collator,
    train_dataset = train_val_dataset['train'],
    eval_dataset = train_val_dataset['test'],
)
trainer.train()
#save model
trainer.save_model(OUTPUT_DIR+"/saved_model")

#prediction
