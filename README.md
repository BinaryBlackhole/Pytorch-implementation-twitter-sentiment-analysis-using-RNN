# Pytorch-implementation-twitter-sentiment-analysis-using-RNN



1. Please download the dataset from this link and keep it in main directory: https://www.kaggle.com/kazanova/sentiment140
2. put the csv file in the working directory and mention the full in the data_config.json
3. in data_config you need to provide :

{
	"dataset_full_path": "twiteer_dataset_main.csv", # full path of the main dataset
	"num_neg_labels": "10000",                       # number of negetive sample the dataset should contain
	"num_pos_labels": "10000",						 # number of positive samples the dataset should contain
	"trainset_fullpath": "data/train.csv",			 # path to save the train.csv	
	"testset_fullpath": "data/test.csv",			 # path to save test.csv	
	"num_training_sample": "15000"					 # number of training samples in train.csv
}

		
4. run python data_processing.py. This will prepare the data by splitting into train and test.
5. To run this project run : Python train.py . To run train.py you need to provide certain params in project_config.py:

{
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
}
to change dataset/any parameters for training: project_config.json

For the limitation of RAM I have taken 20000 samples from the main dataset which achieved 99.23% accuracy in the train dataset.
The test accuracy is `~68% which can be improved by using more data/ training more number of epoch.

References:
https://github.com/akurniawan/pytorch-sentiment-analysis
https://github.com/bentrevett/pytorch-sentiment-analysis
