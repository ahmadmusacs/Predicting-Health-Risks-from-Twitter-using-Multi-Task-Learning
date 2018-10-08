from __future__ import unicode_literals
from __future__ import print_function
from itertools import islice
import io
import nltk
import statistics
from sklearn import svm
import re
import collections
import pickle
import os

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
	with io.open(FilePath, encoding="utf8") as fp:
		for line in fp:
			vocabulary = re.split(r'\t+', line)
		return vocabulary


def BuildingTrainingSetUsingSVM(OutputLabels, featureMatrix, modelName):
	with io.open(OutputLabels) as stateFile:
		# for x in range(0,50):
		# 	print(len(featureMatrix[x]))
		Labels = []

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
			# path = os.getcwd()
			# print ("The current working directory is %s" % path) 
			try:
				os.mkdir("data/" + modelName)
			except OSError:
				print ("")
			# else:
			# 	print ("Successfully created the directory %s " % modelName)

			filename = "data/"+ modelName + "/finalized_model" + str(x) + ".sav"
			joblib.dump(clf, filename)

			
		# print("Done")

def TestingSetUsingSVM(OutputLabels, modelName, Vocabulary, ErrorAnalysisOutputFile, featureMatrix):
	with io.open(OutputLabels) as stateFile,  io.open(ErrorAnalysisOutputFile, 'w') as outputAnalysis:
		correctPrediction  = 0
		Labels = []
		vocabulary = Vocabulary

		for line in stateFile:
			value = line.split()
			Labels.append(int(value[0]))

		for x in range(51):
			filename = 'data/' + modelName+ '/finalized_model' + str(x) + '.sav'
			loaded_model = pickle.load(open(filename, 'rb'))
			predictedLabel = loaded_model.predict([featureMatrix[x]])
			# print(predictedLabel)
			if Labels[x] == predictedLabel[0]:
				correctPrediction = correctPrediction + 1

			weights = loaded_model.coef_[0]
			#print(weights)
			outputAnalysis.write('Predicted Label : {0} , Correct Label : {1} \n'.format(predictedLabel[0] , Labels[x]))
			highest_weighted_idx = sorted(range(len(weights)), key=lambda i: weights[i])[-20:]
			for val in highest_weighted_idx:
				if( val >= len(vocabulary)):
					outputAnalysis.write('{0} {1}\n'.format(str(val-len(vocabulary)+1) , str(weights[val])))
				else:
					outputAnalysis.write('{0} {1}\n'.format(str(vocabulary[val]) , str(weights[val])))
			outputAnalysis.write("................................................\n")
			highest_weighted_idx = sorted(range(len(weights)), key=lambda i: weights[i])[:20]
			for val in highest_weighted_idx:
				if( val >= len(vocabulary)):
					outputAnalysis.write('{0} {1}\n'.format(str(val-len(vocabulary)+1) , str(weights[val])))
				else:
					outputAnalysis.write('{0} {1}\n'.format(str(vocabulary[val]) , str(weights[val])))
			outputAnalysis.write("................................................\n")
			outputAnalysis.write("................................................\n\n")
		print('Number of correct predictions for this set : {0}'.format(correctPrediction))
		return correctPrediction


def main():

	print("Starting Program")
	print("Running Test dataset on trained SVM models [LOOCV] ....")

	
	featureMat = LoadFeatureMatrixFromFile("data/FeatureMatrixAllWords2.txt")
	vocab = BuildVocab("data/AllWords.txt")
	# BuildingTrainingSetUsingSVM("data/obesityBin.txt", featureMat, "allmodelobesity")
	# BuildingTrainingSetUsingSVM("data/DiabetesBin.txt", featureMat, "allmodeldiabetes")
	scoreForAllwords1 = TestingSetUsingSVM("data/obesityBin.txt","allmodelobesity", vocab, "result/TopFeaturesObesityUsingAllWords.txt", featureMat)
	scoreForAllwords2 = TestingSetUsingSVM("data/DiabetesBin.txt", "allmodeldiabetes", vocab, "result/TopFeaturesDiabetesUsingAllWords.txt",  featureMat)
	print("Using All Words as features , following result occurs: ")
	print('Obesity rate {0} , Diabetes Rate {1} '.format(scoreForAllwords1*100/51, scoreForAllwords2*100/51))
	print("-------------------------------------------------------------------------------")

	featureMat = LoadFeatureMatrixFromFile("data/FeatureMatrixAllWordsandLDAupdated2.txt")
	vocab = BuildVocab("data/AllWords.txt")
	# BuildingTrainingSetUsingSVM("data/obesityBin.txt", featureMat, "allmodelLDAobesity")
	# BuildingTrainingSetUsingSVM("data/DiabetesBin.txt", featureMat, "allmodelLDAdiabetes")
	scoreForAllwordsAndLda1 = TestingSetUsingSVM("data/obesityBin.txt","allmodelLDAobesity", vocab, "result/TopFeaturesObesityUsingAllWordsandLDA.txt", featureMat)
	scoreForAllwordsAndLda2 = TestingSetUsingSVM("data/DiabetesBin.txt", "allmodelLDAdiabetes", vocab, "result/TopFeaturesDiabetesUsingAllWordsandLDA.txt",  featureMat)
	print("Using All Words + LDA as features , following result occurs: ")
	print('Obesity rate {0} , Diabetes Rate {1} '.format(scoreForAllwordsAndLda1*100/51, scoreForAllwordsAndLda2*100/51))
	print("-------------------------------------------------------------------------------")

	featureMat = LoadFeatureMatrixFromFile("data/FeatureMatrixFoodWords.txt")
	vocab = BuildVocab("data/FoodWords.txt")
	# BuildingTrainingSetUsingSVM("data/obesityBin.txt", featureMat, "foodmodelobesity")
	# BuildingTrainingSetUsingSVM("data/DiabetesBin.txt", featureMat, "foodmodeldiabetes")
	scoreForFoodwords1 = TestingSetUsingSVM("data/obesityBin.txt","foodmodelobesity", vocab, "result/TopFeaturesObesityUsingFoodWords.txt", featureMat)
	scoreForFoodwords2 = TestingSetUsingSVM("data/DiabetesBin.txt", "foodmodeldiabetes", vocab, "result/TopFeaturesDiabetesUsingFoodWords.txt",  featureMat)
	print("Using Food Words as features , following result occurs: ")
	print('Obesity rate {0} , Diabetes Rate {1} '.format(scoreForFoodwords1*100/51, scoreForFoodwords2*100/51))
	print("-------------------------------------------------------------------------------")


	featureMat = LoadFeatureMatrixFromFile("data/FeatureMatrixFoodWordsandLDAupdated.txt")
	vocab = BuildVocab("data/FoodWords.txt")
	# # # # BuildingTrainingSetUsingSVM("data/obesityBin.txt", featureMat, "foodmodelLDAobesity")
	# # # # BuildingTrainingSetUsingSVM("data/DiabetesBin.txt", featureMat, "foodmodelLDAdiabetes")
	scoreForFoodwordsandLda1 = TestingSetUsingSVM("data/obesityBin.txt","foodmodelLDAobesity", vocab, "result/TopFeaturesObesityUsingFoodWordsandLDA.txt", featureMat)
	scoreForFoodwordsandLda2 = TestingSetUsingSVM("data/DiabetesBin.txt", "foodmodelLDAdiabetes", vocab, "result/TopFeaturesDiabetesUsingFoodWordsandLDA.txt",  featureMat)
	print("Using Food Words+LDA as features , following result occurs: ")
	print('Obesity rate {0} , Diabetes Rate {1} '.format(scoreForFoodwordsandLda1*100/51, scoreForFoodwordsandLda2*100/51))
	print("-------------------------------------------------------------------------------")


	featureMat = LoadFeatureMatrixFromFile("data/FeatureMatrixHashtagWords.txt")
	vocab = BuildVocab("data/HashtagWords.txt")
	# # BuildingTrainingSetUsingSVM("data/obesityBin.txt", featureMat, "hashtagmodelobesity")
	# # BuildingTrainingSetUsingSVM("data/DiabetesBin.txt", featureMat, "hashtagmodeldiabetes")
	scoreForHashtagwords1 = TestingSetUsingSVM("data/obesityBin.txt","hashtagmodelobesity", vocab, "result/TopFeaturesObesityUsingHashtagWords.txt", featureMat)
	scoreForHashtagwords2 = TestingSetUsingSVM("data/DiabetesBin.txt", "hashtagmodeldiabetes", vocab, "result/TopFeaturesDiabetesUsingHashtagWords.txt",  featureMat)
	print("Using Hashtag Words as features , following result occurs: ")
	print('Obesity rate {0} , Diabetes Rate {1} '.format(scoreForHashtagwords1*100/51, scoreForHashtagwords2*100/51))
	print("-------------------------------------------------------------------------------")


	featureMat = LoadFeatureMatrixFromFile("data/FeatureMatrixHashtagWordsandLDAupdated.txt")
	# vocab = BuildVocab("data/HashtagWords.txt")
	# # BuildingTrainingSetUsingSVM("data/obesityBin.txt", featureMat, "hashtagmodelLDAobesity")
	# # BuildingTrainingSetUsingSVM("data/DiabetesBin.txt", featureMat, "hashtagmodelLDAdiabetes")
	scoreForHashtagwordsandLda1 = TestingSetUsingSVM("data/obesityBin.txt","hashtagmodelLDAobesity", vocab, "result/TopFeaturesObesityUsingHashtagWordsandLDA.txt", featureMat)
	scoreForHashtagwordsandLda2 = TestingSetUsingSVM("data/DiabetesBin.txt", "hashtagmodelLDAdiabetes", vocab, "result/TopFeaturesDiabetesUsingHashtagWordsandLDA.txt",  featureMat)
	print("Using Hashtag Words+LDA as features , following result occurs: ")
	print('Obesity rate {0} , Diabetes Rate {1} '.format(scoreForHashtagwordsandLda1*100/51, scoreForHashtagwordsandLda2*100/51))
	print("-------------------------------------------------------------------------------")

	featureMat = LoadFeatureMatrixFromFile("data/FeatureMatrixFoodHashtagWords.txt")
	vocab = BuildVocab("data/HashtagWords.txt")
	FoodandHashtagVocab = vocab + BuildVocab("data/FoodWords.txt")
	# # BuildingTrainingSetUsingSVM("data/obesityBin.txt", featureMat, "foodhashtagmodelobesity")
	# # BuildingTrainingSetUsingSVM("data/DiabetesBin.txt", featureMat, "foodhashtagmodeldiabetes")
	scoreForFoodHashtagwords1 = TestingSetUsingSVM("data/obesityBin.txt","foodhashtagmodelobesity", FoodandHashtagVocab, "result/TopFeaturesObesityUsingFoodHashtagWords.txt", featureMat)
	scoreForFoodHashtagwords2 = TestingSetUsingSVM("data/DiabetesBin.txt", "foodhashtagmodeldiabetes", FoodandHashtagVocab, "result/TopFeaturesDiabetesUsingFoodHashtagWords.txt",  featureMat)
	print("Using Food + Hashtag Words as features , following result occurs: ")
	print('Obesity rate {0} , Diabetes Rate {1} '.format(scoreForFoodHashtagwords1*100/51, scoreForFoodHashtagwords2*100/51))
	print("-------------------------------------------------------------------------------")

	featureMat = LoadFeatureMatrixFromFile("data/FeatureMatrixFoodHashWordsandLDAupdated.txt")
	# vocab = BuildVocab("data/HashtagWords.txt")
	FoodandHashtagVocab = vocab + BuildVocab("data/FoodWords.txt")
	# # BuildingTrainingSetUsingSVM("data/obesityBin.txt", featureMat, "foodhashtagmodelLDAobesity")
	# # BuildingTrainingSetUsingSVM("data/DiabetesBin.txt", featureMat, "foodhashtagmodelLDAdiabetes")
	scoreForFoodHashtagwordsandLda1 = TestingSetUsingSVM("data/obesityBin.txt","foodhashtagmodelLDAobesity", FoodandHashtagVocab, "result/TopFeaturesObesityUsingFoodHashtagWordsandLDA.txt", featureMat)
	scoreForFoodHashtagwordsandLda2 = TestingSetUsingSVM("data/DiabetesBin.txt", "foodhashtagmodelLDAdiabetes", FoodandHashtagVocab, "result/TopFeaturesDiabetesUsingFoodHashtagWordsandLDA.txt",  featureMat)
	print("Using Food + Hashtag Words+ LDA as features , following result occurs: ")
	print('Obesity rate {0} , Diabetes Rate {1} '.format(scoreForFoodHashtagwordsandLda1*100/51, scoreForFoodHashtagwordsandLda2*100/51))
	print("-------------------------------------------------------------------------------")


	print("\nFeatures \t\t\t Obesity Rate \t\t Diabetes Rate\n")
	print("---------------------------------------------------------------------------------\n")
	print("All Words         \t\t {0} \t\t {1}".format(scoreForAllwords1*100/51, scoreForAllwords2*100/51))
	print("All Words + LDA   \t\t {0} \t\t {1}".format(scoreForAllwordsAndLda1*100/51, scoreForAllwordsAndLda2*100/51))
	print("Hashtag Words     \t\t {0} \t\t {1}".format(scoreForHashtagwords1*100/51, scoreForHashtagwords2*100/51))
	print("Hashtag Words + LDA\t\t {0} \t\t {1}".format(scoreForHashtagwordsandLda1*100/51, scoreForHashtagwordsandLda2*100/51))
	print("Food Words        \t\t {0} \t\t {1}".format(scoreForFoodwords1*100/51, scoreForFoodwords2*100/51))
	print("Food Words + LDA   \t\t {0} \t\t {1}".format(scoreForFoodwordsandLda1*100/51, scoreForFoodwordsandLda2*100/51))
	print("Food + Hashtag Words\t\t {0} \t\t {1}".format(scoreForFoodHashtagwords1*100/51, scoreForFoodHashtagwords2*100/51))
	print("Food + Hashtag Words + LDA\t {0} \t\t {1}".format(scoreForFoodHashtagwordsandLda1*100/51, scoreForFoodHashtagwordsandLda2*100/51))

if __name__ == '__main__':
	main()