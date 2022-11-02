import pandas as pd
import re

### KAGGLE_DATASET_URL = https://www.kaggle.com/datasets/basilb2s/language-detection
def remove_non_characters(sentence: str) -> str:
  return re.sub('[0123456789.,";{}+~!@#$%^&*_=|\/<>:?\][-]*', '', sentence)

columns = ["Text", "Language"]
df = pd.read_csv("Language Detection.csv", index_col=False)
df = df[(df['Language'] == 'English') | (df['Language'] == 'Russian')]
df.loc[df['Language'] == 'English', "Language"] = "en"
df.loc[df['Language'] == 'Russian', "Language"] = "ru"
df['Text'] = df['Text'].apply(remove_non_characters)

### HUGGINGFACE_DATASET_URL = https://huggingface.co/datasets/papluca/language-identification
df2 = pd.read_csv("train.csv", index_col=False)
df2 = df2.rename(columns={ "labels": "Language", "text": "Text" })
df2 = df2[(df2['Language'] == "en") | (df2["Language"] == "ru")]
df2 = df2.reindex(columns=columns)

train_df = pd.concat([df, df2])
train_df.to_csv("./cleaned_train.csv", columns=columns, index=False)

df3 = pd.read_csv("test.csv", index_col=False)
df3 = df3.rename(columns={ "labels": "Language", "text": "Text" })
df3 = df3[(df3['Language'] == "en") | (df3["Language"] == "ru")]
df3 = df3.reindex(columns=columns)
df3.to_csv("./cleaned_test.csv", columns=columns, index=False)

df4 = pd.read_csv("validation.csv", index_col=False)
df4 = df4.rename(columns={ "labels": "Language", "text": "Text" })
df4 = df4[(df4['Language'] == "en") | (df4["Language"] == "ru")]
df4 = df4.reindex(columns=columns)
df4.to_csv("./cleaned_validation.csv", columns=columns, index=False)
