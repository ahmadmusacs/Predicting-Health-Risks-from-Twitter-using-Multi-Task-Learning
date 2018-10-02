from __future__ import unicode_literals
from __future__ import print_function
from itertools import islice
import io
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import statistics
from sklearn import svm
import re
import collections



def LoadFeatureMatrixFromFile (FileName):
	with io.open(FileName) as featureMat:
		matrix = [[] for _ in range(56)]
		index = 0
		error = ''
		for line in featureMat:
			valuesList = re.split(r'\t+', line)
			# print(valuesList)
			for x in valuesList:
				# print(x)
				if( x != ' ' and  x != '\n'):
					matrix[index].append(float(x))
			index = index + 1

		return matrix

def BuildVocab(FilePath):
	with io.open(FilePath) as fp:
		for line in fp:
			vocabulary = re.split(r'\t+', line)
		return vocabulary

def TestingDataUsingSVM(OutputLabels, Vocabulary, ErrorAnalysisOutputFile, featureMatrix):
	with io.open(OutputLabels) as stateFile,  io.open(ErrorAnalysisOutputFile, 'w') as outputAnalysis:
		# for x in range(0,50):
		# 	print(len(featureMatrix[x]))
		Labels = []
		

		vocabulary = Vocabulary

		for line in stateFile:
			value = line.split()
			Labels.append(int(value[0]))
			# print(value[0])

		secMat = [[] for _ in range(50)]
		# label = [ 1, 0 ,1, 0]

		correctPrediction = 0
		for x in range(51):
			print(".", end=" ")
			idx = 0
			trainLbl = []
			for y in range(51):
				if x != y:
					secMat[idx] = featureMatrix[y]
					idx = idx + 1
					trainLbl.append(Labels[y])
			# print(secMat)
			# print(trainLbl)
			# print("Training set done")
			clf = svm.LinearSVC()
			clf.fit(secMat, trainLbl)  
			svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
		     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
		     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
		     verbose=0)
			# print("Now testing set")
			# print(Labels[x])
			predictedLabel = clf.predict([featureMatrix[x]]) 
			# print(predictedLabel)
			if Labels[x] == predictedLabel[0]:
				correctPrediction = correctPrediction + 1

			weights = clf.coef_[0]
			#print(weights)
			outputAnalysis.write('Predicted Label : {0} , Correct Label : {1} \n'.format(predictedLabel[0] , Labels[x]))
			highest_weighted_idx = sorted(range(len(weights)), key=lambda i: weights[i])[-20:]
			for val in highest_weighted_idx:
				if( val >= len(vocabulary)):
					outputAnalysis.write('{0} {1}\n'.format(str(val-len(vocabulary)+1) , str(weights[val])))
				else:
					outputAnalysis.write('{0} {1}\n'.format(str(vocabulary[val]) , str(weights[val])))
			
		

		print('\nNumber of correct predictions for this set : {0}\n '.format(correctPrediction))
		return correctPrediction

def main():
	print("Starting Program")
	print("Results sometimes vary in iterations\nTesting SVM classifier for obesity and diabetes rate using all words ....")

	featureMat = LoadFeatureMatrixFromFile("data/FeatureMatrixAllWords2.txt")
	vocab = BuildVocab("data/AllWords.txt")
	numOfCorrectPrediction1 = TestingDataUsingSVM("data/obesityBin.txt", vocab, "ErrorForObesityUsingAllWords.txt", featureMat)
	numOfCorrectPrediction2 = TestingDataUsingSVM("data/DiabetesBin.txt", vocab, "ErrorForDiabetesUsingAllWords.txt", featureMat)
	print("Using All Words as vocab , following result occurs: ")
	print('Obesity rate {0} , Diabetes Rate {1} '.format(numOfCorrectPrediction1*100/51, numOfCorrectPrediction2*100/51))

	
	featureMat = LoadFeatureMatrixFromFile("data/FeatureMatrixAllWordsandLDAupdated2.txt")
	vocab = BuildVocab("data/AllWords.txt")
	numOfCorrectPrediction1 = TestingDataUsingSVM("data/obesityBin.txt", vocab, "ErrorForObesityUsingAllWordsandLDA.txt", featureMat)
	numOfCorrectPrediction2 = TestingDataUsingSVM("data/DiabetesBin.txt", vocab, "ErrorForDiabetesUsingAllWordsandLDA.txt", featureMat)
	print("Using All Words + LDA as vocab , following result occurs: ")
	print('Obesity rate {0} , Diabetes Rate {1} '.format(numOfCorrectPrediction1*100/51, numOfCorrectPrediction2*100/51))

	featureMat = LoadFeatureMatrixFromFile("data/FeatureMatrixFoodWords.txt")
	vocab = BuildVocab("data/FoodWords.txt")
	numOfCorrectPrediction1 = TestingDataUsingSVM("data/obesityBin.txt", vocab, "ErrorForObesityUsingFoodWords.txt", featureMat)
	numOfCorrectPrediction2 = TestingDataUsingSVM("data/DiabetesBin.txt", vocab, "ErrorForDiabetesUsingFoodWords.txt", featureMat)
	print("Using Food Words as vocab , following result occurs: ")
	print('Obesity rate {0} , Diabetes Rate {1} '.format(numOfCorrectPrediction1*100/51, numOfCorrectPrediction2*100/51))


	featureMat = LoadFeatureMatrixFromFile("data/FeatureMatrixFoodWordsandLDAupdated.txt")
	vocab = BuildVocab("data/FoodWords.txt")
	numOfCorrectPrediction1 = TestingDataUsingSVM("data/obesityBin.txt", vocab, "ErrorForObesityUsingFoodWordsandLDA.txt", featureMat)
	numOfCorrectPrediction2 = TestingDataUsingSVM("data/DiabetesBin.txt", vocab, "ErrorForDiabetesUsingFoodWordsandLDA.txt", featureMat)
	print("Using Food Words + LDA as vocab , following result occurs: ")
	print('Obesity rate {0} , Diabetes Rate {1} '.format(numOfCorrectPrediction1*100/51, numOfCorrectPrediction2*100/51))
	

	featureMat = LoadFeatureMatrixFromFile("data/FeatureMatrixHashtagWords.txt")
	vocab = BuildVocab("data/HashtagWords.txt")
	numOfCorrectPrediction1 = TestingDataUsingSVM("data/obesityBin.txt", vocab, "ErrorForObesityUsingHashtagWords.txt", featureMat)
	numOfCorrectPrediction2 = TestingDataUsingSVM("data/DiabetesBin.txt", vocab, "ErrorForDiabetesUsingHashtagWords.txt", featureMat)
	print("Using Hashtag Words as vocab , following result occurs: ")
	print('Obesity rate {0} , Diabetes Rate {1} '.format(numOfCorrectPrediction1*100/51, numOfCorrectPrediction2*100/51))


	featureMat = LoadFeatureMatrixFromFile("data/FeatureMatrixHashtagWordsandLDAupdated.txt")
	vocab = BuildVocab("data/HashtagWords.txt")
	numOfCorrectPrediction1 = TestingDataUsingSVM("data/obesityBin.txt", vocab, "ErrorForObesityUsingHashtagWordsandLDA.txt", featureMat)
	numOfCorrectPrediction2 = TestingDataUsingSVM("data/DiabetesBin.txt", vocab, "ErrorForDiabetesUsingHashtagWordsandLDA.txt", featureMat)
	print("Using Hashtag Words + LDA as vocab , following result occurs: ")
	print('Obesity rate {0} , Diabetes Rate {1} '.format(numOfCorrectPrediction1*100/51, numOfCorrectPrediction2*100/51))


	featureMat = LoadFeatureMatrixFromFile("data/FeatureMatrixFoodHashtagWords.txt")
	vocab = BuildVocab("data/HashtagWords.txt")
	FoodandHashtagVocab = vocab + BuildVocab("data/FoodWords.txt")
	numOfCorrectPrediction1 = TestingDataUsingSVM("data/obesityBin.txt", FoodandHashtagVocab, "ErrorForObesityUsingFoodHashtagWords.txt", featureMat)
	numOfCorrectPrediction2 = TestingDataUsingSVM("data/DiabetesBin.txt", FoodandHashtagVocab, "ErrorForDiabetesUsingFoodHashtagWords.txt", featureMat)
	print("Using Food + hashtag Words as vocab , following result occurs: ")
	print('Obesity rate {0} , Diabetes Rate {1} '.format(numOfCorrectPrediction1*100/51, numOfCorrectPrediction2*100/51))

	featureMat = LoadFeatureMatrixFromFile("data/FeatureMatrixFoodHashWordsandLDAupdated.txt")
	vocab = BuildVocab("data/HashtagWords.txt")
	FoodandHashtagVocab = vocab + BuildVocab("data/FoodWords.txt")
	numOfCorrectPrediction1 = TestingDataUsingSVM("data/obesityBin.txt", FoodandHashtagVocab, "ErrorForObesityUsingFoodHashtagWordsandLDA.txt", featureMat)
	numOfCorrectPrediction2 = TestingDataUsingSVM("data/DiabetesBin.txt", FoodandHashtagVocab, "ErrorForDiabetesUsingFoodHashtagWordsandLDA.txt", featureMat)
	print("Using Food + hashtag Words + LDA as vocab , following result occurs: ")
	print('Obesity rate {0} , Diabetes Rate {1} '.format(numOfCorrectPrediction1*100/51, numOfCorrectPrediction2*100/51))



if __name__ == '__main__':
	main()