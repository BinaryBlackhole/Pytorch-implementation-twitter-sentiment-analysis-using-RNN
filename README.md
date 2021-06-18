<a id="requirements"></a>
# Twitter Sentiment Analysis Using RNN - PyTorch


[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" />](https://pytorch.org/)
[<img alt="Pandas" src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" />](https://pandas.pydata.org/)
[<img alt="NumPy" src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" />](https://numpy.org/)
[<img alt="Kaggle" src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white" />](https://www.kaggle.com/kazanova/sentiment140)

[![Python 3.6.8](https://img.shields.io/badge/python-3.6.8-blue.svg)](https://www.python.org/downloads/release/python-368/)
[![torch 1.8.1](https://img.shields.io/badge/torch-1.8.1-orange.svg)](https://pypi.org/project/torch/1.8.1/)
[![torch_text 0.9.1](https://img.shields.io/badge/torchtext-0.9.1-orange.svg)](https://pypi.org/project/torchtext/0.9.1/)
[![pytorch_ignite 0.4.4](https://img.shields.io/badge/pytorch--ignite-0.4.4-orange.svg)](https://pypi.org/project/pytorch-ignite/0.4.4/) 


[![Stargazers repo roster for @BinaryBlackhole/Pytorch-implementation-twitter-sentiment-analysis-using-RNN](https://reporoster.com/stars/BinaryBlackhole/Pytorch-implementation-twitter-sentiment-analysis-using-RNN)](https://github.com/BinaryBlackhole/Pytorch-implementation-twitter-sentiment-analysis-using-RNN/stargazers)
[![Forkers repo roster for @BinaryBlackhole/Pytorch-implementation-twitter-sentiment-analysis-using-RNN](https://reporoster.com/forks/BinaryBlackhole/Pytorch-implementation-twitter-sentiment-analysis-using-RNN)](https://github.com/BinaryBlackhole/Pytorch-implementation-twitter-sentiment-analysis-using-RNN/network/members)
## Introduction

![intro_image](https://i.morioh.com/2020/02/04/beef36fd707d.jpg) \
Image Source: morioh

The automated process of recognition and Categorization of instinctive information in text script is called Sentiment Analysis. And such categorization of positive tweets from negative tweets by machine learning models for Classification, text mining, text analysis and data visualization through Natural Language processing is called Twitter Sentiment Analysis.

![project_workflow_example](https://user-images.githubusercontent.com/49767657/121781346-dbb30000-cbc1-11eb-809a-a016d7a6092f.png) \
Image Source: Google

## Requirements

This project is developed on [<img alt="Windows 10" src="https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white" />](https://www.microsoft.com/en-in/software-download/windows10) \
Clone this repository or download after installing the above [requirements](#requirements) and run `pip install -r requirement.txt` to install all the libraries required to run this project.
You can also click on the specific badges mentioned [above](#requirements) and download it as well.


- **torch==1.8.1+cu** 
- **torchtext==0.9.1** 
- **pytorch-ignite==0.4.4** 
- **pandas==1.0.5** 
- **numpy==1.19.3** 
### IDE
I have used [<img alt="PyCharm" src="https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green"/>](https://www.jetbrains.com/pycharm/) in this project. I mostly prefer using Pycharm for building large end-to-end projects.
Though for data analytics & visualizations I would recommend of using [<img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white"/>](https://jupyter.org/). If you don't know how to use jupyter which is quite easy for beginners, go through this [link](https://www.tutorialspoint.com/jupyter/index.htm)

	


## To Run this Project
1. Download the dataset from [kaggle](#requirements) above and keep it in main directory. 
2. Put the csv file in the working directory and mention the full in the **data_config.json**
3. In **data_config.json** you need to provide :

	i) `dataset_full_path`: **full path of the main dataset** \
	ii) `num_neg_labels`: **number of negetive sample the dataset should contain** \
	iii)`num_pos_labels`: **number of positive samples the dataset should contain** \
	iv) `trainset_fullpath`: **path to save the train.csv** \
	v)  `testset_fullpath`: **path to save test.csv**      \
	vi) `num_training_sample`: **number of training samples in train.csv** 
		
4. *run* `python data_processing.py`: **This will prepare the data by splitting into train and test.**
5. To run this project,you need to provide certain parameters in **project_config.json**:

	i) `data_path`: **data** \
	ii)`model_dir`: **saved_model** \
	iii) `device`: **-1** \
	iv) `model_name`: **sentiment_classifer_rnn_sagar.pt** \
	v) `embedding_dim`: **100** \
	vi) `hidden_dim`: **256** \
	vii) `output_dim`: **1** \
	viii) `batch_size`: **64** \
	ix) `max_vocab_size`: **25000** \
	x)  `learning_rate`: <img src="http://www.sciweavers.org/tex2img.php?eq=%201e%5E%7B-3%7D%20&bc=White&fc=Black&im=jpg&fs=18&ff=ccfonts,eulervm&edit=0" align="center" border="0" alt=" 1e^{-3} " width="52" height="25" /> \
	xi) `epoch`: **20** 
6. After performing the above steps Execute : `Python train.py`

**To change dataset/any parameters for training: project_config.json**

## Experiment Details

- epochs : **20**
- Optimizer- **Adam**
- Learning rate:<img src="http://www.sciweavers.org/tex2img.php?eq=%201e%5E%7B-3%7D%20&bc=White&fc=Black&im=jpg&fs=18&ff=ccfonts,eulervm&edit=0" align="center" border="0" alt=" 1e^{-3} " width="52" height="25" /> 
- Loss function : **BCElogisticloss**
- Train Acc: **99.72**
- Validation accuracy: **67.68**
- Test accuracy: **64.54**
- Number of training examples: **10500** 
- Number of validation examples: **4500 Number of testing examples: 3962**
- Unique tokens in TEXT vocabulary: **18609**
- Unique tokens in LABEL vocabulary: **2**
- The model has **1,952,805** trainable parameters

For the limitation of RAM I have taken **20000 samples** from the main dataset which achieved **99.23% accuracy** in the train dataset.

The test accuracy is `68%` which can be improved by using more data training more number of epoch.

## Authors
#### Sagar Chakraborty 
[<img alt="Sagar_Chakraborty_LinkedIn" src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />](https://www.linkedin.com/in/binaryblackhole/)
[<img alt="Sagar_Chakraborty_Gmail" src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" />](https://mail.google.com/mail/u/0/#search/csagar963%40gmail.com)
[<img alt="Sagar_Chakraborty_GitHub" src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white" />](https://github.com/BinaryBlackhole)
#### Akshata Kulkarni
[<img alt="Akshata_Kulkarni_LinkedIn" src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />](https://www.linkedin.com/in/akshata-kulkarni-3a0005161/)
## Thanks to ALL the amazing contributors!
![GitHub Contributors Image](https://contrib.rocks/image?repo=BinaryBlackhole/Pytorch-implementation-twitter-sentiment-analysis-using-RNN)

## References
#### Akurniawan-sentiment analysis
[<img alt="Akurniawan_sentiment_analysis_github" src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" />](https://github.com/akurniawan/pytorch-sentiment-analysis) 
#### Bentrevett-sentiment analysis
[<img alt="Bentrevett_sentiment_analsis" src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" />](https://github.com/bentrevett/pytorch-sentiment-analysis)

**If you like this project please fork and star this project**


![open_source](https://forthebadge.com/images/badges/open-source.svg) 
[![Built-With-Love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/Naereen/) \
[![MIT_Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge)](./LICENSE)

