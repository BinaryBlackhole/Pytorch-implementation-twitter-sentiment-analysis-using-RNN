# Pytorch-implementation-twitter-sentiment-analysis-using-RNN

## Introduction
The automated process of recognition and Categorization of instinctive information in text script is called Sentiment Analysis. And such categorization of positive tweets from negative tweets by machine learning models for Classification, text mining, text analysis and data visualization through Natural Language processing is called Twitter Sentiment Analysis.
![image taken from google](https://user-images.githubusercontent.com/49767657/121781346-dbb30000-cbc1-11eb-809a-a016d7a6092f.png)
Image Source: Google

## Requirements
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
This project was developed on Windows 10 using [Python 3.6.8](https://www.python.org/downloads/) \
Clone this repository or download the codes after installing python and run **pip install -r requirement.txt** to install all the libraries required to run this project.

- **torch==1.8.1+cpu** 
- **torchtext==0.9.1** 
- **pytorch-ignite==0.4.4** 
- **pandas==1.0.5** 
- **numpy==1.19.3** 
	


## To Run this Project
1. Please download the dataset from this link and keep it in main directory: [Twitter Sentiment Dataset](https://www.kaggle.com/kazanova/sentiment140)
2. Put the csv file in the working directory and mention the full in the data_config.json
3. in **data_config.json** you need to provide :

	i) *"dataset_full_path":(# full path of the main dataset)* \
	ii) *"num_neg_labels": (# number of negetive sample the dataset should contain)* \
	iii)*"num_pos_labels": (# number of positive samples the dataset should contain)* \
	iv) *"trainset_fullpath": (# path to save the train.csv)* \
	v)  *"testset_fullpath": (# path to save test.csv)*      \
	vi) *"num_training_sample": (# number of training samples in train.csv)* 
		
4. run python data_processing.py. This will prepare the data by splitting into train and test.
5. To run this project run : Python train.py . To run train.py -\
   you need to provide certain params in **project_config.json**:

	i) *"data_path": "data",* \
	ii)*"model_dir": "saved_model",* \
	iii) *"device": "-1",* \
	iv) *"model_name": "sentiment_classifer_rnn_sagar.pt",* \
	v) *"embedding_dim": "100",* \
	vi) *"hidden_dim": "256",* \
	vii) *"output_dim": "1",* \
	viii) *"batch_size": "64",* \
	ix) *"max_vocab_size": "25000",* \
	x)" *learning_rate": "1e-3",* \
	xi) *"epoch": "20"*

**To change dataset/any parameters for training: project_config.json**

## Experiment Details

- epochs : 20
- Optimizer- Adam Learning rate: 1e-3 
- Loss function : BCElogisticloss
- Train Acc: 99.72
- Validation accuracy: 67.68
- Test accuracy: 64.54 
- Number of training examples: 10500 
- Number of validation examples: 4500 Number of testing examples: 3962
- Unique tokens in TEXT vocabulary: 18609
- Unique tokens in LABEL vocabulary: 2
- The model has 1,952,805 trainable parameters

For the limitation of RAM I have taken **20000 samples** from the main dataset which achieved **99.23% accuracy** in the train dataset.
The test accuracy is `68% which can be improved by using more data training more number of epoch.

## Contributors
1. [Sagar Chakraborty](https://www.linkedin.com/in/binaryblackhole/)
2. [Akshata Kulkarni](https://www.linkedin.com/in/akshata-kulkarni-3a0005161/)

## References
1. [Akurniawan-sentiment analysis](https://github.com/akurniawan/pytorch-sentiment-analysis) 
2. [Bentrevett-sentiment analysis](https://github.com/bentrevett/pytorch-sentiment-analysis)


[![GitHub forks](https://img.shields.io/github/forks/BinaryBlackhole/Pytorch-implementation-twitter-sentiment-analysis-using-RNN)](https://github.com/BinaryBlackhole/Pytorch-implementation-twitter-sentiment-analysis-using-RNN/network)
[![GitHub stars](https://img.shields.io/github/stars/BinaryBlackhole/Pytorch-implementation-twitter-sentiment-analysis-using-RNN)](https://github.com/BinaryBlackhole/Pytorch-implementation-twitter-sentiment-analysis-using-RNN/stargazers)
