
# Predicting Individual Diabetes Risk from Tweets

## Our Approach:
This is a project for CSC585. We tried to predict diabetes risk for individuals from tweet texts. We proposed a CNN with different class weights for positive and negative samples. The CNN architecture has 3 convolutional layer where each word embeddings are feed into. 
The file ``my_script`` contains the source code to train and test. 

### **Tested Environment**
```
Docker ( CPU )
	OS: Windows 10
	Lib: Python 3.7
Host:
	OS: Windows 10
	CPU: Intel Core i3-3110M @ 2.40 GHz
	RAM: 6 GB
```

### **Installation**

At first, use git to clone the project. `git clone https://github.com/ahmadmusacs/Predicting-Health-Risks-from-Twitter-using-Multi-Task-Learning.git`. Then, `cd` into the project. 

Now, run this docker command to build a docker image for the project. I could not make spacy tokenizer working with docker. So I had to remove them out and the scores dorpped a lot for that reason. So, sorry for the mismatch with the paper result. 
```
>> docker build -t twitter .
``` 
There are two iterations in this project. 
To run iteration 1 
```
>> docker run twitter 1
```

To run iteration 2
```
>> docker run twitter 2
```



## Baseline implementation using SVM
This is a replication of the work from this [paper](https://arxiv.org/pdf/1409.2195). I implemented the baseline for predicting health risks such as: obesity rate and overweight rate by analyzing the tweet data. [Scikit Learn SVM](http://scikit-learn.org/stable/modules/svm.html) has been used for classification tasks. In order to run the project using Python, user has to install scikit learn and nltk. But using the Dockerfile is more convenient as it takes care of the dependencies. 

The folder ``data`` contains all the models that were trained with SVM. The file ``main.py`` contains all the relevant codes to train and test using SVM. The ``result`` folder has the top weighted features which SVM learns during the training phase for both classes. 

### **Tested Environment**
```
Docker ( CPU )
	OS: Windows 10
	Lib: Python 3.7
Host:
	OS: Windows 10
	CPU: Intel Core i3-3110M @ 2.40 GHz
	RAM: 6 GB
```

### **Installation**

At first, use git to clone the project. `git clone https://github.com/ahmadmusacs/Predicting-Health-Risks-from-Twitter-using-Multi-Task-Learning.git`. Then, `cd` into the project. 

Now, run this docker command to build a docker image for the project. 
```
>> docker build -t twitter .
``` 
After this step is completed, run the following command. 
```
>> docker run twitter
```

### **Description of the output**

The output of the program will show you the accuracy of the models on the prediction tasks. 
Below, it is a sample sanp from the output. It takes less than 20 minutes to perform classification tasks on the test dataset. 

![Image of sample run 2](https://github.com/ahmadmusacs/Predicting-Health-Risks-from-Twitter-using-Multi-Task-Learning/blob/master/images/final_result.JPG)
