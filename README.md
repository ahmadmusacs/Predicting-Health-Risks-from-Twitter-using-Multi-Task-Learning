# Predicting Health Risks from Twitter using Multi Task Learning

This is a replication of the work from this [paper](https://arxiv.org/pdf/1409.2195). I implemented the baseline for predicting health risks such as: obesity rate and overweight rate by analyzing the tweet data. [Scikit Learn SVM](http://scikit-learn.org/stable/modules/svm.html) has been used for classification tasks. In order to run the project using Python, user has to install scikit learn and nltk. But using the Dockerfile is more convenient as it takes care of the dependencies. 


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

At first, use git to clone the project. `git clone https://github.com/ahmadmusacs/Predicting-Health-Risks-from-Twitter-using-Multi-Task-Learning.git`. 

Now, run this docker command to build docker image. 
```
	>> docker build -t twitter4food .
``` 
After this step is completed, run docker using this command. 
```
	>> docker run --name twitter4food
```

### **Discription of the output**

The output of the program will show you the accuracy of the models on the prediction tasks. 
Below, it is a sample sanp from the output. 
![Image of sample run] (https://github.com/ahmadmusacs/Predicting-Health-Risks-from-Twitter-using-Multi-Task-Learning/blob/master/images/outputSnap.JPG)

