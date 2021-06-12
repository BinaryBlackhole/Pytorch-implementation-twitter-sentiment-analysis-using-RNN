# Pytorch-implementation-twitter-sentiment-analysis-using-RNN
The automated process of recognition and Categorization of instinctive information in text script is called Sentiment Analysis. And such categorization of positive tweets from negative tweets by machine learning models for Classification, text mining, text analysis and data visualization through Natural Language processing is called Twitter Sentiment Analysis.
![image](https://user-images.githubusercontent.com/49767657/121781346-dbb30000-cbc1-11eb-809a-a016d7a6092f.png)

#To Run this Project
1. Please download the dataset from this link and keep it in main directory: https://www.kaggle.com/kazanova/sentiment140
2. put the csv file in the working directory and mention the full in the data_config.json
3. in data_config you need to provide :

"dataset_full_path": "twiteer_dataset_main.csv", (# full path of the main dataset) \n
"num_neg_labels": "10000",                       (# number of negetive sample the dataset should contain) \n
"num_pos_labels": "10000",			(# number of positive samples the dataset should contain) \n
"trainset_fullpath": "data/train.csv",		(# path to save the train.csv) \n
"testset_fullpath": "data/test.csv",		(# path to save test.csv)      \n
"num_training_sample": "15000"			(# number of training samples in train.csv) \n
		
4. run python data_processing.py. This will prepare the data by splitting into train and test.
5. To run this project run : Python train.py . To run train.py you need to provide certain params in project_config.py:


"data_path": "data",
"model_dir": "saved_model",
"device": "-1",
"model_name": "sentiment_classifer_rnn_sagar.pt",
"embedding_dim": "100",
"hidden_dim": "256",
"output_dim": "1",
"batch_size": "64",
"max_vocab_size": "25000",
"learning_rate": "1e-3",
"epoch": "20"

#to change dataset/any parameters for training: project_config.json

For the limitation of RAM I have taken 20000 samples from the main dataset which achieved 99.23% accuracy in the train dataset.
The test accuracy is `~68% which can be improved by using more data/ training more number of epoch.

#Contributors:
1. Sagar Chakraborty
2. Akshata Kulkarni

#References:
https://github.com/akurniawan/pytorch-sentiment-analysis
https://github.com/bentrevett/pytorch-sentiment-analysis
