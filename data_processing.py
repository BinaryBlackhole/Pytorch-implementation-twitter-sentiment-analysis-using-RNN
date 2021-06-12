"Author: Sagar Chakraborty"
import pandas as pd
import numpy as np
import json


# parsing the data_config.json to get the params
"""
{
	"dataset_full_path": "twiteer_dataset_main.csv",
	"num_neg_labels": "10000",
	"num_pos_labels": "10000",
	"trainset_fullpath": "data/train.csv",
	"testset_fullpath": "data/test.csv",
	"num_training_sample": "15000"
}"""


f = open('data_config.json','r')
config_data = json.loads(f.read())

dataset= config_data['dataset_full_path']
neg_labels = int(config_data['num_neg_labels'])
pos_labels = int(config_data['num_pos_labels'])
train_filepath = config_data['trainset_fullpath']
test_filepath = config_data['testset_fullpath']
training_samples = int(config_data['num_training_sample'])






dataset_df  = pd.read_csv(dataset,names=['labels','id','datetime','query','username','sentences'],header= None,sep=',',encoding = "ISO-8859-1")
#
print(dataset_df['labels'].head(5))

# if dataset needs to cut short here we took 10000 for each class
filter_0_df = dataset_df[dataset_df['labels']== int(0)].sample(n=neg_labels)
filter_4_df = dataset_df[dataset_df['labels']== int(4)].sample(n=pos_labels)


# dropping unneccessary columns
filter_0_df.drop(['id','datetime','query','username'], axis=1,inplace=True)
filter_4_df.drop(['id','datetime','query','username'],axis=1,inplace=True)

print(filter_0_df)
print(filter_4_df)



# merging datasframe of two class to construct final dataframe
final_df = pd.concat([filter_0_df,filter_4_df])

print(len(final_df))
#drop the rows wherever we have links/url in the tweets mostly spam
final_df= final_df.drop(final_df[final_df.sentences.str.contains(r'http\S+|www.\S+')].index)

#shuffle the dataset
final_df=final_df.sample(n= len(final_df), random_state=42)

# dirty way to convert labels in 0 and 1
final_df[final_df['labels']>0]=1
print(len(final_df))

# removing the headers
final_df= final_df[1:]


# train_test split
train_df= final_df[:training_samples]
test_df= final_df[training_samples:]

#saving data
train_df.to_csv(train_filepath)
test_df.to_csv(test_filepath)

