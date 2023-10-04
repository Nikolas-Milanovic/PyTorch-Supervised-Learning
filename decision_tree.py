import pandas as pd
import numpy as np
import scipy.stats
import copy


################# entropoy
# Tip: feel free to call scipy.stats.entropy.  Make sure to use log base 2.
#
# Input:
# data_frame -- pandas data frame
# 
# Output:
# answer -- float indicating the empirical entropy of the data in data_frame
##############################################
def entropy(data_frame):
	# count = data_frame['class'].value_counts()
	p = len(data_frame[data_frame['class'] == 'high'])
	n = len(data_frame[data_frame['class'] == 'low'])
	total = p + n
	# prob_dist = np.array([(count['high']/count.sum()),(count['low']/count.sum())])
	prob_dist = np.array([(p/(total)),(n/(total))])
	answer = scipy.stats.entropy(prob_dist, base=2)
	return answer
    

################# info_gain
# Tip: this function should call entropy
#
# Inputs:
# data_frame -- pandas data frame
# attribute -- string indicating the attribute for which we wish to compute the information gain
# domain -- set of values (strings) that the attribute can take
#
# Output:
# answer -- float indicating the information gain
################################################3
def info_gain(data_frame, attribute, domain):
	p = len(data_frame[data_frame['class'] == 'high'])
	n = len(data_frame[data_frame['class'] == 'low'])
	remainder = 0 
	for value in domain:
		p_i =  len(data_frame[(data_frame[attribute] == value) & (data_frame['class'] == 'high')])
		n_i =  len(data_frame[(data_frame[attribute] == value) & (data_frame['class'] == 'low')])
		total_i = p_i+n_i
		if(total_i==0):
			continue
		prob_dist = np.array([(p_i/(total_i)),(n_i/(total_i))])
		remainder+= (total_i/(p+n))*(scipy.stats.entropy(prob_dist, base=2))
		
	answer = entropy(data_frame) - remainder
	return answer

######## Decision_tree class
#
# This class defines the data structure of the decision tree to be learnt
############################
class Decision_Tree:

	# constructor
	def __init__(self,attribute,branches,label):
		self.attribute = attribute
		self.branches = branches
		self.label = label

	# leaf constructor
	def make_leaf(label):
		return Decision_Tree('class', {}, label)
	
	# node constructor
	def make_node(attribute,branches):
		return Decision_Tree(attribute, branches, None)

	# string representation
	def __repr__(self):
		return self.string_repr(0)
		
	# decision tree string representation
	def string_repr(self,indent):
		indentation = '\t'*indent
		
		# leaf string representation
		if self.attribute == 'class':
			return f'\n{indentation}class = {self.label}'

		# node string representation
		else:
			representation = ''
			for value in self.branches:
				representation += f'\n{indentation}{self.attribute} = {value}:'
				representation += self.branches[value].string_repr(indent+1)
			return representation

	# classify a data point
	def classify(self,data_point):

		# leaf
		if self.attribute == 'class':
			return self.label

		# node
		else:
			return self.branches[data_point[self.attribute]].classify(data_point)

############# choose attribute
# Tip: this function should call info_gain
#
# Inputs:
# attributes_with_domains -- dictionary with attributes as keys and domains as values
# data_frame -- pandas data_frame
#
# Output:
# best_score -- float indicting the information gain score of the best attribute
# best_attribute -- string indicating the best attribute
#################################
def choose_attribute(attributes_with_domains: dict(),data_frame):
	best_score = 0
	best_attribute = 'NULL'

	for attribute, domain in attributes_with_domains.items():
		IG = info_gain(data_frame,attribute,domain)
		if( IG > best_score):
			best_score = IG
			best_attribute = attribute
	
	return best_score, best_attribute

############# train decision tree
# Tip: this is a recursive function that should call itself as well as
# choose_attribute,  Decision_Tree.make_leaf, Decision_Tree.make_node
#
# Inputs:
# data_frame -- pandas data frame
# attributes_with_domains -- dictionary with attributes as keys and domains as values
# default_class -- string indicating the class to be assigned when data_frame is empty
# threshold -- integer indicating the minimum number of data points in data_frame to allow
#              the creation of a new node that splits the data with some attribute
#
# Output:
# decision_tree -- Decision_Tree object
########################
def train_decision_tree(data_frame,attributes_with_domains:dict(),default_class,threshold):
	if(len(data_frame)==0): 
		return Decision_Tree.make_leaf(default_class)
	elif(len(data_frame)<threshold): 
		value = data_frame['class'].mode()[0]
		return Decision_Tree.make_leaf(value)
	elif(entropy(data_frame)==0):
		value = data_frame.loc[data_frame['class'].first_valid_index(), 'class']
		return Decision_Tree.make_leaf(value)
	elif(len(attributes_with_domains)==0):
		value = data_frame['class'].mode()[0]
		return Decision_Tree.make_leaf(default_class)
	else:
		bestScore, bestAttr = choose_attribute(attributes_with_domains,data_frame)
		tree = Decision_Tree.make_node(bestAttr,{})
		for value in attributes_with_domains[bestAttr]:
			data_frame_copy = copy.deepcopy(data_frame)
			attributes_with_domains_copy = copy.deepcopy(attributes_with_domains)
			attributes_with_domains_copy.pop(bestAttr)
			subtree = train_decision_tree(data_frame_copy[data_frame_copy[bestAttr] == value],
				 attributes_with_domains_copy,
				 data_frame['class'].mode()[0],
				 threshold)
			tree.branches[value]=subtree
		return tree

	

######### eval decision tree
# Tip: this function should call decision_tree.classify
#
# Inputs:
# decision tree -- Decision_Tree object
# data_frame -- pandas data frame
#
# Output:
# accuracy -- float indicating the accuracy of the decision tree
#############
def eval_decision_tree(decision_tree: Decision_Tree, data_frame):
	correct = 0
	for index, row in data_frame.iterrows():
		if(row['class'] == decision_tree.classify(row)):
			correct+=1
	accuracy = correct/len(data_frame)
	return accuracy

########### k-fold cross-validation
# Tip: this function should call train_decision_tree and eval_decision_tree
#
# Inputs:
# train_data -- pandas data frame
# test_data -- pandas data frame
# attributes_with_domains -- dictionary with attributes as keys and domains as values
# k -- integer indicating the number of folds
# threshold_list -- list of thresholds to be evaluated
#
# Outputs:
# best_threshold -- integer indiating the best threshold found by cross validation
# test_accuracy -- float indicating the accuracy based on the test set
#####################################3
def cross_validation(train_data, test_data, attributes_with_domains, k, threshold_list):
	accuracy = 0
	best_accuracy = 0 
	best_threshold = threshold_list[0]
	best_k = 0 

	num_rows = len(train_data)
	fold_size = num_rows // k
	for threshold in threshold_list:
		accuracy = 0
		for i in range(k):
			validationSet = train_data[i*fold_size:(i+1)*fold_size]
			trainSet = pd.concat([train_data[:i*fold_size], train_data[(i+1)*fold_size:]], axis=0)
			#print(i*fold_size,(i+1)*fold_size, len(validationSet))
			decisionTree = train_decision_tree(trainSet,
				      attributes_with_domains,
				      trainSet['class'].mode()[0],
					  threshold)
			if(i==0):
				print(decisionTree)

			#print(decisionTree)
			accuracy += eval_decision_tree(decisionTree,validationSet)
		accuracy = accuracy/k
		print(f'Threshold {threshold}: average validation accuracy {accuracy}')
		if(accuracy > best_accuracy):
			best_accuracy = accuracy
			best_threshold = threshold
			best_k = i 

	#best_trainSet = pd.concat([train_data[:best_k*fold_size], train_data[(best_k+1)*fold_size:]], axis=0)
	best_trainSet = train_data
	best_tree = train_decision_tree(best_trainSet,
				      attributes_with_domains,
				      best_trainSet['class'].mode()[0],
					  best_threshold)
	test_accuracy = eval_decision_tree(best_tree,test_data)
	print(best_tree)
		
	return best_threshold, test_accuracy

############################ main
# You should not need to change the code below
#
# This code performs the following operations:
# 1) Load the data
# 2) create a list of attributes
# 3) create a dictionary that maps each attribute to its domain of values
# 4) split the data into train and test sets
# 5) train a decision tree while optimizing the threshold hyperparameter by
#    10-fold cross validation 
#####################################

#load data
data_frame = pd.read_csv("categorical_real_estate.csv")
data_frame = data_frame.fillna('NA')
print(data_frame)

# get attributes
attributes = list(data_frame.columns)
attributes.remove('class')

# create dictionary that maps each attribute to its domain of values
attributes_with_domains = {}
for attr in attributes:
	attributes_with_domains[attr] = set(data_frame[attr])


#split data in to train and test
train_data = data_frame.iloc[0:1000]
test_data = data_frame.iloc[1000:]

# perform 10-fold cross-validation
best_threshold, accuracy = cross_validation(train_data, test_data, attributes_with_domains, 10, [10,20,40,80,160])
print(f'Best threshold {best_threshold}: accuracy {accuracy}')






