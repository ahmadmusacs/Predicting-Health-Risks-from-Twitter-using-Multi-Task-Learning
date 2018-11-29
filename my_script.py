from __future__ import unicode_literals
from __future__ import print_function
from itertools import islice
import io
import statistics
import re
import collections
import pickle
import os
import csv
import torch
from torchtext.data import TabularDataset
from torchtext.data import Field
import json
from torchtext import data
import jsonlines
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import spacy
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import codecs
import pickle
import sys

countItr = 0
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv_0 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[0],embedding_dim))
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[1],embedding_dim))
        self.conv_2 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[2],embedding_dim))
        self.fc = nn.Linear(3 * len(filter_sizes)*n_filters, 2)
        self.fc1 = nn.Linear(len(filter_sizes)*n_filters, len(filter_sizes)*n_filters)
        self.fc2 = nn.Linear(len(filter_sizes)*n_filters, len(filter_sizes)*n_filters)
        self.fc3 = nn.Linear(len(filter_sizes)*n_filters, len(filter_sizes)*n_filters)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, y):
        
        #x = [sent len, batch size]
        
        x = x.permute(1, 0)
                
        #x = [batch size, sent len]
        
        embedded = self.embedding(x)
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
            
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))

        #cat = [batch size, n_filters * len(filter_sizes)]

        datax = self.fc1(cat)
        sumtensorval = y.sum().data[0]
        emptyZeros = torch.zeros(cat.size())

        if( sumtensorval == 0):
        	datay = self.fc2(cat)
        	dataz = self.fc3(emptyZeros)
        	cat = torch.cat((datax , datay, dataz), dim=1)
        else:
        	datay = self.fc3(cat)
        	dataz = self.fc2(emptyZeros)
        	cat = torch.cat((datax , datay, dataz), dim=1)

        

        return self.fc(cat)

class CNNWithoutDomainAdaptation(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv_0 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[0],embedding_dim))
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[1],embedding_dim))
        self.conv_2 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[2],embedding_dim))
        self.fc = nn.Linear( len(filter_sizes)*n_filters, 2)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [sent len, batch size]
        
        x = x.permute(1, 0)
                
        #x = [batch size, sent len]
        
        embedded = self.embedding(x)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
            
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))

        #cat = [batch size, n_filters * len(filter_sizes)]       

        return self.fc(cat)


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):

        #x = [sent len, batch size]
        
        embedded = self.embedding(x)
        
        #embedded = [sent len, batch size, emb dim]
        
        output, hidden = self.rnn(embedded)
        
        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        
        assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        
        return self.fc(hidden.squeeze(0))

def mergeDataset(dataset1 , dataset2):
	
	data = []
	with codecs.open(dataset1,'rU','utf-8') as f , codecs.open(dataset2, 'rU' , 'utf8') as f2:
		for line in f:
			data.append(json.loads(line))
		for line in f2:
			data.append(json.loads(line))
		with jsonlines.open("TrainSetNew.jsonl", mode='w') as writer:
			writer.write_all(data)
			writer.close()
		
def addDomainLabel(dataset1, filename):
	listOfData = []
	data = {}
	with codecs.open(dataset1,'rU','utf-8') as f:
		index = 0
		for line in f:
			index += 1
			# if index <= 450:
			# 	data = json.loads(line)
			# 	data["domain"] = "statedata"
			# else:
			data = json.loads(line)
			data["domain"] = "indidata"
			listOfData.append(data)

		with jsonlines.open(filename, mode='w') as writer:
				writer.write_all(listOfData)
				writer.close()


def mergeDatasetForStateTask(dataset1 , dataset2, dataset3):
	
	data = []
	testData = []
	with codecs.open(dataset1,'rU','utf-8') as f , codecs.open(dataset2, 'rU' , 'utf8') as f2, codecs.open(dataset3, 'rU' , 'utf8') as f3:
		index = 0
		for line in f:
			index += 1
			if( index >= 26):
				testData.append(json.loads(line))
			else:
				data.append(json.loads(line))

		for line in f2:
			data.append(json.loads(line))
		for line in f3:
			data.append(json.loads(line))

		with jsonlines.open("TrainDataStateTask.jsonl", mode='w') as writer:
			writer.write_all(data)
			writer.close()
		with jsonlines.open("TestDataStateTask.jsonl", mode='w') as writer:
			writer.write_all(testData)
			writer.close()


def loadDataset(datasetFile, outputDatasetFile, individualDatasetFile):
	with io.open("DiabetesBin.txt") as labelFile, io.open(datasetFile, encoding="utf8") as dataset, io.open(individualDatasetFile
		, encoding="utf8") as indDs:
		
		# outputWriter = csv.writer(output, delimiter='\t')
		index = 0
		labels = []
		
		listOfdict = []
		for line in labelFile:
			# print(line)
			if(line == "1\n"):
				print("true")
				labels.append("pos")
			else:
				print("false")
				labels.append("neg")
		
		print(str("len of labels ") + str(len(labels)))
		# outputWriter.writerow(["tweet", "label"])
		
		for line in dataset:
			data = {}
			print(index)
			new_string = ''
			
			# new_string += '\n'
			# final_string = re.findall(r'\w+', new_string)
			# print(new_string)
			word_counts = Counter(word for word in line.split())
			
			for i in line.split():
				# print(i)
				if (word_counts[i]>1 and i != '\n'):
					if i[:1] == '@':
						pass
					elif i[:1] == '#':
						new_string = new_string + ' ' + i[1:]
					elif i.find("http") >= 0:
						pass
					else:
						new_string = new_string + ' ' + i
			data["tweet"] = new_string
			print("one state done : size of string -> " + str(len(new_string)))
			data["label"] = labels[index]
			listOfdict.append(data)
			# outputWriter.writerow([new_string, labels[index]])
			index = index + 1
			# index = index + 1
		print(len(listOfdict))
		# print(listOfdict[0])
		# print(listOfdict[1])

		for itrx in range(51):
			newList = []
			for itry in range(51):
				if itrx != itry:
					newList.append(listOfdict[itry])
			saveToDisk("/" + outputDatasetFile+str(itrx)+str(".jsonl"))
			saveToDisk("/" + outputDatasetFile+str(itrx)+str(".jsonl"))
			
			with jsonlines.open(outputDatasetFile+str(itrx)+str(".jsonl"), mode='w') as writer:
				writer.write_all(newList)
				writer.close()
			
			
		

		# jstr = json.dump(listOfdict, output, ensure_ascii=False)
		# print(jstr)
def saveToDisk(filename):
    
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
        	print("what the hell!")
    else:
    	print("already exists")

def BuildIndividualDataset(individualDatasetFile):
	with io.open(individualDatasetFile, encoding="utf8") as dataset:
		
		# outputWriter = csv.writer(output, delimiter='\t')
		index = 0
		labels = []
		
		listOfdict = []
		
		# outputWriter.writerow(["tweet", "label"])
		numberOfDataPoints = 603
		lineNumber = 0
		skip = 0
		targetLineNumber = 2
		CollectionOfTweets = []
		
		tweet = ''
		currentLabel = ''
		for line in dataset:
			lineNumber += 1
			if( lineNumber == targetLineNumber):
				skip = 0
			# print("Line number " + str(lineNumber))
			if lineNumber == 1:
				continue
			if skip == 2:
				lineStringSet = line.split()
				print(lineStringSet)
				targetLineNumber = int(lineStringSet[len(lineStringSet)-1])*2 + 3 + lineNumber
				print("target number of tweets " + str(targetLineNumber))
				skip -= 1
				continue
			if skip == 1:
				skip -= 1
				continue

			lineStringSet = line.split()
			# print(lineStringSet)
			if( lineNumber == targetLineNumber):
				print("I am in 1")
				print(lineStringSet)
				labels.append(lineStringSet[1])
				if tweet != '':
					# CollectionOfTweets.append(tweet)
					data = {}
					data["tweet"] = tweet
					if currentLabel == 'not':
						currentLabel = 'neg'
					else:
						currentLabel = 'pos'
					data["label"] = currentLabel
					print("adding previos label " + currentLabel)
					listOfdict.append(data)
				currentLabel = lineStringSet[1]
				skip = 2
				print(lineStringSet[0] + " " + str(len(tweet)))
				
				tweet = ''
				continue
			
			# word_counts = Counter(word for word in lineStringSet)
			
			for word in lineStringSet:
				if '<@MENTION>' in word or '<URL>' in word or '<NUMBER>' in word:
					# print("passing")
					continue
				if word == '\n':
					continue
				if word[:1] == '#':
					word = word[1:]

				tweet = tweet + ' ' + word
			# print(tweet)
			
			skip = 1

			# print(numberOfTweets)
			new_string = ''
		print(len(listOfdict))
		trainset = []
		testset = []
		for x in range(450):
			trainset.append(listOfdict[x])
		for x in range(451,600):
			testset.append(listOfdict[x])
		with jsonlines.open("TrainSet.jsonl", mode='w') as writer:
				writer.write_all(trainset)
				writer.close()
		with jsonlines.open("TestSet.jsonl", mode='w') as writer:
				writer.write_all(testset)
				writer.close()


def IndividualPredictionTask(pretrainedModel, trainDataset, testDataset, labels):
	TEXT = data.Field(tokenize='spacy')
	LABEL = labels

	
	#TEXT = Field(tokenize=tokenize, lower=True,tensor_type=torch.cuda.LongTensor)
	#LABEL = data.LabelField(dtype=torch.float, tensor_type=torch.cuda.LongTensor)

	fields = {'tweet': ('text', TEXT), 'label': ('label', LABEL)}
	train_data, test_data = data.TabularDataset.splits(
                            path = '~/',
                            train = trainDataset,
                            test = testDataset,
                            format = 'json',
                            fields = fields)

	

	print("Vocab build done")
	TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
	
	print(LABEL.vocab.stoi)
	print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
	print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
	print(TEXT.vocab.freqs.most_common(20))

	#print(TEXT.vocab)

	BATCH_SIZE = 5

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	#print("device available " + device)

	train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
		(train_data, train_data, test_data), sort_key=lambda x: len(x.text),
		batch_size=BATCH_SIZE,
		device=device)


	model = pretrainedModel
	optimizer = optim.Adam(model.parameters())

	criterion = nn.BCEWithLogitsLoss()

	model = model.to(device)
	criterion = criterion.to(device)

	N_EPOCHS = 7

	for epoch in range(N_EPOCHS):

	    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
	    #valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
	    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% ')
	    #print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')


	#

	test_loss, test_acc = evaluate(model, test_iterator, criterion)

	print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')  



def BuildEmbeddings(trainset, testset):

	SEED = 1234

	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.backends.cudnn.deterministic = True

	TEXT = data.Field(tokenize='spacy')
	LABEL = data.LabelField(dtype=torch.float)

	tokenize = lambda x: x.split()
	#TEXT = Field(tokenize=tokenize, lower=True,tensor_type=torch.cuda.LongTensor)
	#LABEL = data.LabelField(dtype=torch.float, tensor_type=torch.cuda.LongTensor)

	fields = {'tweet': ('text', TEXT), 'label': ('label', LABEL)}
	train_data, test_data = data.TabularDataset.splits(
                            path = '~/',
                            train = trainset,
                            test = testset,
                            format = 'json',
                            fields = fields)

	print("Done")

	
	TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
	LABEL.build_vocab(train_data)
	print(LABEL.vocab.stoi)

	print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
	print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
	print(TEXT.vocab.freqs.most_common(20))

	#print(TEXT.vocab)

	BATCH_SIZE = 10

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	#print("device available " + device)

	train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
		(train_data, train_data, test_data), sort_key=lambda x: len(x.text),
		batch_size=BATCH_SIZE,
		device=device)

	INPUT_DIM = len(TEXT.vocab)
	EMBEDDING_DIM = 100
	N_FILTERS = 100
	FILTER_SIZES = [3,4,5]
	OUTPUT_DIM = 1
	DROPOUT = 0.5

	model = CNNWithoutDomainAdaptation(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
	pretrained_embeddings = TEXT.vocab.vectors

	model.embedding.weight.data.copy_(pretrained_embeddings)
	
	optimizer = optim.Adam(model.parameters())
	weights = [1, 4]
	class_weights = torch.FloatTensor(weights)
	criterion = nn.CrossEntropyLoss(weight=class_weights, reduce=False)

	# criterion = nn.BCEWithLogitsLoss()

	model = model.to(device)
	criterion = criterion.to(device)

	print("Model's state_dict:")
	for param_tensor in model.state_dict():
		print(param_tensor, "\t", model.state_dict()[param_tensor].size())

	#saveToDisk("./model1.pt")
	#saveToDisk("./finalNetwork2.pt")

	
	#pickle.dump( model.state_dict(), open( "saveModel.pt", "wb" ) )
	#model.load_state_dict(torch.load("saveModel.pt"))
	#for param_tensor in model.state_dict():
	#	print(param_tensor, "\t", model.state_dict()[param_tensor].size())

	N_EPOCHS = 5

	# IndividualPredictionTask(model, "individualDatasetFresh.jsonl", "individualDatasetFresh.jsonl")

	for epoch in range(N_EPOCHS):

	    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
	    #valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
	    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% ')
	    #print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')
	


	#IndividualPredictionTask(model, "TrainSet.jsonl", "TestSet.jsonl", LABEL)

	torch.save(model.state_dict(), './TrainedModelOnFood4W.pt')

	test_loss, test_acc, test_precision, test_recall = evaluate(model, test_iterator, criterion)
	test_f1_score = 2*(test_precision * test_recall) / (test_precision+test_recall)
	print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% | Test Precision: {test_precision:.2f}%  | Test Recall: {test_recall:.2f}%')   
	print(f'| Test F1 Score : {test_f1_score:.2f}')
	

	
def TestDomainAdaptation(trainset, testset):
	SEED = 1234

	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.backends.cudnn.deterministic = True

	TEXT = data.Field(tokenize='spacy')
	LABEL = data.LabelField(dtype=torch.float)
	DOMAIN = data.LabelField(dtype=torch.float)

	tokenize = lambda x: x.split()
	#TEXT = Field(tokenize=tokenize, lower=True,tensor_type=torch.cuda.LongTensor)
	#LABEL = data.LabelField(dtype=torch.float, tensor_type=torch.cuda.LongTensor)

	fields = {'tweet': ('text', TEXT), 'label': ('label', LABEL), 'domain': ('domain', DOMAIN)}
	train_data, test_data = data.TabularDataset.splits(
                            path = '~/',
                            train = trainset,
                            test = testset,
                            format = 'json',
                            fields = fields)

	print("Done")

	
	TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
	LABEL.build_vocab(train_data)
	DOMAIN.build_vocab(train_data)
	print(DOMAIN.vocab.stoi)
	print(LABEL.vocab.stoi)
	print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
	print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
	print(f"Unique tokens in LABEL vocabulary: {len(DOMAIN.vocab)}")
	print(TEXT.vocab.freqs.most_common(20))

	#print(TEXT.vocab)

	BATCH_SIZE = 5

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	#print("device available " + device)

	train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
		(train_data, train_data, test_data), sort_key=lambda x: len(x.text),
		batch_size=BATCH_SIZE,
		device=device)

	INPUT_DIM = len(TEXT.vocab)
	EMBEDDING_DIM = 100
	N_FILTERS = 100
	FILTER_SIZES = [3,4,5]
	OUTPUT_DIM = 1
	DROPOUT = 0.5

	model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
	pretrained_embeddings = TEXT.vocab.vectors

	model.embedding.weight.data.copy_(pretrained_embeddings)
	
	optimizer = optim.Adam(model.parameters())

	# criterion = nn.BCEWithLogitsLoss()
	weights = [0.3, 0.7]
	class_weights = torch.FloatTensor(weights)
	criterion = nn.CrossEntropyLoss(weight=class_weights, reduce=False)

	model = model.to(device)
	criterion = criterion.to(device)

	print("Model's state_dict:")
	for param_tensor in model.state_dict():
		print(param_tensor, "\t", model.state_dict()[param_tensor].size())

	#saveToDisk("./model1.pt")
	#saveToDisk("./finalNetwork2.pt")

	
	#pickle.dump( model.state_dict(), open( "saveModel.pt", "wb" ) )
	#model.load_state_dict(torch.load("saveModel.pt"))
	#for param_tensor in model.state_dict():
	#	print(param_tensor, "\t", model.state_dict()[param_tensor].size())

	N_EPOCHS = 5

	# IndividualPredictionTask(model, "individualDatasetFresh.jsonl", "individualDatasetFresh.jsonl")

	for epoch in range(N_EPOCHS):

	    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
	    #valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
	    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% ')
	    #print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')
	


	#IndividualPredictionTask(model, "TrainSet.jsonl", "TestSet.jsonl", LABEL)

	torch.save(model.state_dict(), './TrainedModelOnDomainwithWeightLoss.pt')

	test_loss, test_acc = evaluate(model, test_iterator, criterion)

	print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')  
	
	# listchildren = list(model.children())[1].parameters()
	# for x in listchildren:
	# 	print(x)

	# mod = list(model.children())
	# mod.pop()
	# mod.append(torch.nn.Linear(len(filter_sizes)*n_filters, output_dim))
	# new_classifier = torch.nn.Sequential(*mod)
	# listchildren = list(new_classifier.children())[1].parameters()
	# for x in listchildren:
	# 	print(x)
	# model = new_classifier

	# INPUT_DIM = len(TEXT.vocab)

	# EMBEDDING_DIM = 100
	# HIDDEN_DIM = 256
	# OUTPUT_DIM = 1

	# model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
	# optimizer = optim.SGD(model.parameters(), lr=1e-3)
	# criterion = nn.BCEWithLogitsLoss()
	
	# model = model.to(device)
	# criterion = criterion.to(device)
	# N_EPOCHS = 5

	# for epoch in range(N_EPOCHS):

	#     train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
	    
	#     print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% ')

countItr = 0
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    
    #print(y)
    #round predictions to the closest integer
    
    temp_preds = nn.functional.softmax(preds, dim=1)
    global countItr
    
    rounded_preds = torch.argmax(temp_preds, dim=1)
    # print(temp_preds)
    # print(rounded_preds)
    # print(y)
    # print("----")
    # print(rounded_preds)
    # print(y)
    truePositive = 0
    falsePositive = 0
    for x in range(len(y)):
    	if( rounded_preds[x] == 1 and y[x] == 1 ):
    		truePositive += 1
    	elif( rounded_preds[x] == 1 and y[x] == 0):
    		falsePositive += 1

    countItr += 1
    #print(y)
    correct = (rounded_preds == y).float() #convert into float for division 
    #print(correct)
    #print(rounded_preds)
    #print(correct.sum())
    #print(len(correct))
    #print("------------")

    acc = correct.sum()/len(correct)
    return acc


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_precision = 0
    epoch_recall = 0
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            batchlabel = batch.label.long()

            loss = criterion(predictions, batchlabel)
            #print(predictions)
            acc,precision, recall = binary_accuracy_modified(predictions, batchlabel)
	        #precision = calc_precision(predictions, batch.label)
        	#recall = calc_recall(predictions, batch.label)
            epoch_loss += loss.sum().item()
            epoch_acc += acc.item()
            # print(len(iterator))
            epoch_precision += precision
            epoch_recall += recall
            #epoch_prec += precision.item()
			#epoch_recall += recall.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_precision/ len(iterator), epoch_recall/len(iterator)

truePositive = 0
falsePositive = 0
falseNegative = 0
trueNegative = 0
def binary_accuracy_modified(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    
    #print(y)
    #round predictions to the closest integer
    
    temp_preds = nn.functional.softmax(preds, dim=1)
    global countItr
    
    rounded_preds = torch.argmax(temp_preds, dim=1)
    # print(temp_preds)
    # print(rounded_preds)
    # print(y)
    # print("----")
    global truePositive
    global trueNegative
    global falsePositive
    global falseNegative
    
    one = torch.ones(1, dtype=torch.long)
    zero = torch.zeros(1, dtype=torch.long)
    # print(one[0])

    for x in range(len(rounded_preds)):
    	# print( str(rounded_preds[x]) + " " + str(y[x]))
    	if( rounded_preds[x] == y[x]  and y[x] == one[0] ):
    		truePositive += 1
    	elif( rounded_preds[x] == one[0] and y[x] == zero[0]):
    		falsePositive += 1
    	elif( rounded_preds[x] != y[x] and y[x] == one[0]):
    		falseNegative += 1
    	elif( rounded_preds[x] == y[x] and y[x] == zero[0]):
    		trueNegative += 1

    	#print(str(truePositive) + " " + str(falsePositive))

    precision = 100
    recall = 100
    if(truePositive + falseNegative)> 0 :
    	recall = 100 * truePositive / ( truePositive + falseNegative)

    if (truePositive+falsePositive)>0 :
	    precision = 100 * truePositive / (truePositive + falsePositive)
	
	# recall = 100 * truePositive / (truePositive + falseNegative)
    # print("precision : " + str(precision))
    #print(y)
    correct = (rounded_preds == y).float() #convert into float for division 
    #print(correct)
    #print(rounded_preds)
    #print(correct.sum())
    #print(len(correct))
    #print("------------")

    acc = correct.sum()/len(correct)
    # print(f'true +ve {truePositive:.2f} true -ve {trueNegative:.2f} false +ve {falsePositive:.2f} false -ve {falseNegative:.2f}')
    countItr += 1
    return acc, precision, recall

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        # print(len(batch.text))
        batchlabel = batch.label.long()
        # print(predictions)
        # print(batchlabel)
        loss = criterion(predictions, batchlabel)

        # print(loss)
        # print(batch.label)
        # print("-------------")

        acc = binary_accuracy(predictions, batchlabel)

        # loss.backward()
        loss.sum().backward()        
        optimizer.step()
        
        epoch_loss += loss.sum().item()
        epoch_acc += acc.item()
        
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate_modified(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_precision = 0
    epoch_recall = 0
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text, batch.domain).squeeze(1)
            # print(predictions)
            batchlabel = batch.label.long()
            loss = criterion(predictions, batchlabel)
            
            acc,precision, recall = binary_accuracy_modified(loss, batch.label)
	        #precision = calc_precision(predictions, batch.label)
        	#recall = calc_recall(predictions, batch.label)
            epoch_loss += loss.sum().item()
            epoch_acc += acc.item()
            # print(len(iterator))
            epoch_precision += precision
            epoch_recall += recall
            # print(recall)
            #epoch_prec += precision.item()
			#epoch_recall += recall.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_precision/ len(iterator), epoch_recall/len(iterator)

# def evaluate(model, iterator, criterion):
    
#     epoch_loss = 0
#     epoch_acc = 0
    
#     model.eval()
    
#     with torch.no_grad():
    
#         for batch in iterator:

#             predictions = model(batch.text).squeeze(1)
            
#             loss = criterion(predictions, batch.label)
            
#             acc = binary_accuracy(predictions, batch.label)

#             epoch_loss += loss.item()
#             epoch_acc += acc.item()
        
#     return epoch_loss / len(iterator), epoch_acc / len(iterator)


# def train(model, iterator, optimizer, criterion):
    
#     epoch_loss = 0
#     epoch_acc = 0
    
#     model.train()
    
#     for batch in iterator:
        
#         # print(batch)

#         optimizer.zero_grad()
                
#         predictions = model(batch.text).squeeze(1)
        
#         # print(predictions)
#         # print(batch.label)

#         loss = criterion(predictions, batch.label)
        
#         acc = binary_accuracy(predictions, batch.label)
        
#         loss.backward()
        
#         optimizer.step()
        
#         epoch_loss += loss.item()
#         epoch_acc += acc.item()
        
#     return epoch_loss / len(iterator), epoch_acc / len(iterator)

# def binary_accuracy(preds, y):
#     """
#     Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
#     """

#     #round predictions to the closest integer
#     rounded_preds = torch.round(torch.sigmoid(preds))
#     correct = (rounded_preds == y).float() #convert into float for division 
#     acc = correct.sum()/len(correct)
#     return acc


def RunIteration2(trainset, testset, modelfile):
	
	SEED = 1234

	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.backends.cudnn.deterministic = True

	TEXT = data.Field(tokenize='spacy')
	LABEL = data.LabelField(dtype=torch.float)
	#DOMAIN = data.LabelField(dtype=torch.float)

	tokenize = lambda x: x.split()
	#TEXT = Field(tokenize=tokenize, lower=True,tensor_type=torch.cuda.LongTensor)
	#LABEL = data.LabelField(dtype=torch.float, tensor_type=torch.cuda.LongTensor)

	fields = {'tweet': ('text', TEXT), 'label': ('label', LABEL)}
	# fields = {'tweet': ('text', TEXT), 'label': ('label', LABEL), 'domain': ('domain', DOMAIN)}
	train_data, test_data = data.TabularDataset.splits(
                            path = '~/',
                            train = trainset,
                            test = testset,
                            format = 'json',
                            fields = fields)

	# print("Done")

	
	TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
	LABEL.build_vocab(train_data)
	#DOMAIN.build_vocab(train_data)
	# print(LABEL.vocab.stoi)

	# print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
	# print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
	# print(TEXT.vocab.freqs.most_common(20))

	#print(TEXT.vocab)

	BATCH_SIZE = 5

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	#print("device available " + device)

	train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
		(train_data, train_data, test_data), sort_key=lambda x: len(x.text),
		batch_size=BATCH_SIZE,
		device=device)
	INPUT_DIM = len(TEXT.vocab)
	EMBEDDING_DIM = 100
	N_FILTERS = 100
	FILTER_SIZES = [3,4,5]
	OUTPUT_DIM = 1
	DROPOUT = 0.5
	# print(len(test_data))

	model = CNNWithoutDomainAdaptation(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)

	model.load_state_dict(torch.load(modelfile))

	#model.load_state_dict(torch.load('./TrainedModelOnBothDataBCE1.pt'))

	optimizer = optim.Adam(model.parameters())

	# criterion = nn.BCEWithLogitsLoss()
	weights = [0.3, 0.7]
	class_weights = torch.FloatTensor(weights)
	criterion = nn.CrossEntropyLoss(weight=class_weights, reduce=False)

	test_loss, test_acc, test_precision, test_recall = evaluate(model, test_iterator, criterion)
	test_f1_score = 2*(test_precision * test_recall) / (test_precision+test_recall)
	print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% | Test Precision: {test_precision:.2f}%  | Test Recall: {test_recall:.2f}%')   
	print(f'| Test F1 Score : {test_f1_score:.2f}')


def RunIteration1(trainset, testset):
	
	SEED = 1234

	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.backends.cudnn.deterministic = True

	TEXT = data.Field(tokenize='spacy')
	LABEL = data.LabelField(dtype=torch.float)
	DOMAIN = data.LabelField(dtype=torch.float)

	tokenize = lambda x: x.split()
	#TEXT = Field(tokenize=tokenize, lower=True,tensor_type=torch.cuda.LongTensor)
	#LABEL = data.LabelField(dtype=torch.float, tensor_type=torch.cuda.LongTensor)

	#fields = {'tweet': ('text', TEXT), 'label': ('label', LABEL)}
	fields = {'tweet': ('text', TEXT), 'label': ('label', LABEL), 'domain': ('domain', DOMAIN)}
	train_data, test_data = data.TabularDataset.splits(
                            path = '~/',
                            train = trainset,
                            test = testset,
                            format = 'json',
                            fields = fields)

	# print("Done")

	
	TEXT.build_vocab(test_data, max_size=25000, vectors="glove.6B.100d")
	LABEL.build_vocab(test_data)
	DOMAIN.build_vocab(test_data)
	# print(LABEL.vocab.stoi)

	# print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
	# print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
	# print(TEXT.vocab.freqs.most_common(20))

	#print(TEXT.vocab)

	BATCH_SIZE = 5

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	#print("device available " + device)

	train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
		(train_data, train_data, test_data), sort_key=lambda x: len(x.text),
		batch_size=BATCH_SIZE,
		device=device)
	INPUT_DIM = len(TEXT.vocab)
	EMBEDDING_DIM = 100
	N_FILTERS = 100
	FILTER_SIZES = [3,4,5]
	OUTPUT_DIM = 1
	DROPOUT = 0.5



	model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)

	model.load_state_dict(torch.load('./TrainedModelOnDomainwithWeightLoss.pt'))

	optimizer = optim.Adam(model.parameters())

	# criterion = nn.BCEWithLogitsLoss()
	weights = [0.3, 0.7]
	class_weights = torch.FloatTensor(weights)
	criterion = nn.CrossEntropyLoss(weight=class_weights, reduce=False)

	test_loss, test_acc, test_precision, test_recall = evaluate_modified(model, test_iterator, criterion)
	test_f1_score = 2*(test_precision * test_recall) / (test_precision+test_recall)
	print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% | Test Precision: {test_precision:.2f}%  | Test Recall: {test_recall:.2f}%')   
	print(f'| Test F1 Score : {test_f1_score:.2f}')
	print(f'true +ve {truePositive:.2f} true -ve {trueNegative:.2f} false +ve {falsePositive:.2f} false -ve {falseNegative:.2f}')

def MakeNewTrainSet(trainset, testset):
	trainData = []
	testData = []
	with codecs.open(trainset,'rU','utf-8') as f:
		idx = 0
		for line in f:
			idx += 1
			if idx <= 300:
				trainData.append(json.loads(line))
			else:
				testData.append(json.loads(line))
	with codecs.open(testset,'rU','utf-8') as f:
		for line in f:
			trainData.append(json.loads(line))

	with jsonlines.open("TrainSetFiltered4.jsonl", mode='w') as writer:
		writer.write_all(trainData)
		writer.close()
	with jsonlines.open("TestSetFiltered4.jsonl", mode='w') as writer:
		writer.write_all(testData)
		writer.close()
	

def checkData(trainset , testset):
	traindata = []
	testdata = []
	with codecs.open(trainset,'rU','utf-8') as f , codecs.open(testset, 'rU' , 'utf8') as f2:
		for line in f:
			traindata.append(json.loads(line))
		for line in f2:
			testdata.append(json.loads(line))

		for x in range(0 , len(traindata)):
			xdata = traindata[x]
			for y in range(0, len(testdata)):
				ydata = testdata[y]
				#print(f'{len(xdata["tweet"])}  {len(ydata["tweet"])} ')

				if xdata["tweet"] == ydata["tweet"]:
					print("Error! Match Found!")
				elif len(xdata["tweet"]) == len(ydata["tweet"]):
					print("Length matched!")



def main(argv):
	# loadDataset("TweetsforStates.txt", "StateDataJsonLines")
	#BuildEmbeddings("DatasetJsonLines.jsonl")
	# for itrx in range(51):
	#BuildIndividualDataset("tweets_tokenized.txt")
	#mergeDataset("DatasetJsonLines1.jsonl", "TrainSet.jsonl")
	#mergeDatasetForStateTask("DatasetJsonLines1.jsonl", "TrainSet.jsonl" , "TestSet.jsonl")
	# BuildEmbeddings("TrainSetNew.jsonl", "TestSet.jsonl")
	countItr = 0
	#RunIteration2("TrainSetNew.jsonl", "TestSet.jsonl")
	#print(countItr)
	#addDomainLabel("TestSet.jsonl", "TestSetDomain.jsonl")
	#TestDomainAdaptation("TrainSetDomain.jsonl", "TestSetDomain.jsonl")
	#RunIteration1("TrainSetDomain.jsonl", "TestSetDomain.jsonl")
	
	option = int(argv[0])
	if( option == 1):
		print("Running fold 1")
		RunIteration2("/TrainSetNew2.jsonl", "/TestSet.jsonl", "/TrainedModelOnBothDataBCE1.pt")
		print("Running fold 2")
		RunIteration2("/TrainSetNew3.jsonl", "/TestSet3.jsonl", "/TrainedModelOnBothDataBCE2.pt")
		print("Running fold 3")
		RunIteration2("/TrainSetNew4.jsonl", "/TestSet4.jsonl", "/TrainedModelOnBothDataBCE3.pt")
		print("Running fold 4")
		RunIteration2("/TrainSetNew5.jsonl", "/TestSet5.jsonl", "/TrainedModelOnBothDataBCE4.pt")
		
	elif( option == 2 ):
		print("Running fold 1")
		RunIteration2("/TrainSetFiltered.jsonl", "/TestSetFiltered.jsonl", "/TrainedModelOnFood1W.pt")
		print("Running fold 2")
		RunIteration2("/TrainSetFiltered2.jsonl", "/TestSetFiltered2.jsonl", "/TrainedModelOnFood1W.pt")
		print("Running fold 3")
		RunIteration2("/TrainSetFiltered3.jsonl", "/TestSetFiltered3.jsonl", "/TrainedModelOnFood1W.pt")
		print("Running fold 4")
		RunIteration2("/TrainSetFiltered4.jsonl", "/TestSetFiltered4.jsonl", "/TrainedModelOnFood1W.pt")

countItr = 0
if __name__ == '__main__':
	main(sys.argv[1:])
