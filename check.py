import pandas as pd

df = pd.read_csv("data/train.csv")

lang = int(len(df)/3)

df1 = df.iloc[:,:lang]
df2 = df.iloc[:,lang:]

df1.to_csv("train1.csv")
df2.to_csv("valid.csv")

!python convert_tf_checkpoint_to_pytorch.py
from pytorch_pretrained_bert.modeling import BertPreTrainedModel




















#######################
