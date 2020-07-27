import os
import random
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from nltk.tag import pos_tag
import re
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import gensim 
from gensim.models import Word2Vec
from sklearn.neighbors import KNeighborsClassifier
import sys


def NormalizeWithPOS(text):
	# Lemmatization & Stemming according to POS tagging

	word_list = word_tokenize(text)
	rev = []
	lemmatizer = WordNetLemmatizer() 
	stemmer = PorterStemmer() 
	for word, tag in pos_tag(word_list):
		if tag.startswith('J'):
			w = lemmatizer.lemmatize(word, pos='a')
		elif tag.startswith('V'):
			w = lemmatizer.lemmatize(word, pos='v')
		elif tag.startswith('N'):
			w = lemmatizer.lemmatize(word, pos='n')
		elif tag.startswith('R'):
			w = lemmatizer.lemmatize(word, pos='r')
		else:
			w = word
		w = stemmer.stem(w)
		rev.append(w)
	review = ' '.join(rev)
	return review
	
def cleanText(text):
	
	text = re.sub(r'<.*?>', ' ', text)
	text = re.sub(r"won't", "will not", text)
	text = re.sub(r"can't", "can not", text)
	text = re.sub(r"n't", " not", text)
	text = re.sub(r"'ve", " have", text)
	text = re.sub(r"'ll", " will", text)
	text = re.sub(r"'re", " are", text)

	if embedding is not 'BERT':
		text = re.sub(r"[0-9]+", ' ', text)
		text = re.sub(r"-", ' ', text)
	
	
	text = text.strip().lower()
	
	if embedding is 'WORD2VEC_NO_STOP':
		# Remove Stop words
		default_stop_words = set(stopwords.words('english'))
		default_stop_words.difference_update({'no', 'not', 'nor', 'too', 'any'})
		stop_words = default_stop_words.union({"'m", "n't", "'d", "'re", "'s",
											   'would','must',"'ve","'ll",'may'})
	
		word_list = word_tokenize(text)
		filtered_list = [w for w in word_list if not w in stop_words]
		text = ' '.join(filtered_list)
	
	if embedding is not 'BERT':
		# Remove other contractions
		text = re.sub(r"'", ' ', text)
	
	# Replace punctuations with space
	if embedding is 'BERT': # save ! ? . for end of the sentence detection [,/():;']
		filters='"#$%&*+<=>@[\\]^_`{|}~\t\n'
		text = re.sub(r'\!+', '!', text)
		text = re.sub(r'\?+', '?', text)
	else:
		filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
	translate_dict = dict((i, " ") for i in filters)
	translate_map = str.maketrans(translate_dict)
	text = text.translate(translate_map)
	
	if embedding is 'BERT':
		text = re.sub(r'\( *\)', ' ', text)

	if embedding is not 'BERT':
		text = ' '.join([w for w in text.split() if len(w)>1])

	# Replace multiple space with one space
	text = re.sub(' +', ' ', text)
	
	text = ''.join(text)

	return text

def mean(z): # used for BERT (word version) and Word2Vec
	return sum(itertools.chain(z))/len(z)
	
def embeddToWord2Vec(text):
	words = word_tokenize(text)
	
	if embedding is 'WORD2VEC_WITH_STOP':
		result = [w2v_with_stop_model.wv[w] for w in words if w in w2v_with_stop_model.wv.vocab]
	else:
		result = [w2v_no_stop_model.wv[w] for w in words if w in w2v_no_stop_model.wv.vocab]
	
	feature = [mean(x) for x in zip(*result)]
	return feature
	
def wordTokenize(text):
	return word_tokenize(text)
	
def embeddToBERT(text):
	sentences = re.split('!|\?|\.',text)
	sentences = list(filter(None, sentences)) 

	if bert_version == 'WORD':
		result = bert(sentences, 'avg') # avg is refer to handle OOV
	
		bert_vocabs_of_sentence = []
		for sentence in range(len(result)):
			for word in range(len(result[sentence][1])):
				bert_vocabs_of_sentence.append(result[sentence][1][word])
		feature = [mean(x) for x in zip(*bert_vocabs_of_sentence)]

	elif bert_version == 'SENTENCE':
		result = bert_transformers.encode(sentences)
		feature = [mean(x) for x in zip(*result)]
  
	return feature
	
# A function to plot and print result
def printResult(y_pred, y_prob, title):
	savepath =	'output/plots/comparative_crossval/'
	acc = accuracy_score(test_data["gold_label"], y_pred)
	f1 = f1_score(test_data["gold_label"], y_pred, average='macro')
	precision = precision_score(test_data["gold_label"], y_pred, average='macro')
	recall = recall_score(test_data["gold_label"], y_pred, average='macro')
	# Result
	print("Accuracy: {:.2f}".format(acc*100),end='\n\n')
	cm = confusion_matrix(test_data["gold_label"],y_pred)
	print('Confusion Matrix:\n', cm)
	print(classification_report(test_data["gold_label"],y_pred))

	# Plot
	fpr, tpr, thresholds = roc_curve(test_data["gold_label"], y_prob)
	roc_auc = auc(fpr, tpr)
	print ("Area under the ROC curve : %f" % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.plot(fpr, tpr, color='red', label='ROC curve (area = %0.2f)' % roc_auc)
	plt.legend(loc='lower right')
	plt.savefig(title+'.png')
	plt.close()
	return f1, precision, recall
	
if __name__ == '__main__':
	sys.stdout = open('output/logs/log.txt', 'w') # redirect prints to log file
	path = "data/train.csv"
	corpus = pd.read_csv(path, encoding='utf-8')
	# cross validation stratified splits into train and test
	kf = KFold(n_splits=3)
	split_count = 1



	knn_bow_cvscores_f1 = []
	knn_bow_cvscores_precision = []
	knn_bow_cvscores_recall = []

	knn_tf_stop_cvscores_f1 = []
	knn_tf_stop_cvscores_precision = []
	knn_tf_stop_cvscores_recall = []

	knn_tf_nostop_cvscores_f1 = []
	knn_tf_nostop_cvscores_precision = []
	knn_tf_nostop_cvscores_recall = []

	knn_w2v_stop_cvscores_f1 = []
	knn_w2v_stop_cvscores_precision = []
	knn_w2v_stop_cvscores_recall = []

	knn_w2v_nostop_cvscores_f1 = []
	knn_w2v_nostop_cvscores_precision = []
	knn_w2v_nostop_cvscores_recall = []

	knn_bert_cvscores_f1 = []
	knn_bert_cvscores_precision = []
	knn_bert_cvscores_recall = []



	svm_bow_cvscores_f1 = []
	svm_bow_cvscores_precision = []
	svm_bow_cvscores_recall = []
	
	svm_tf_stop_cvscores_f1 = []
	svm_tf_stop_cvscores_precision = []
	svm_tf_stop_cvscores_recall = []
	
	svm_tf_nostop_cvscores_f1 = []
	svm_tf_nostop_cvscores_precision = []
	svm_tf_nostop_cvscores_recall = []
	
	svm_w2v_stop_cvscores_f1 = []
	svm_w2v_stop_cvscores_precision = []
	svm_w2v_stop_cvscores_recall = []
	
	svm_w2v_nostop_cvscores_f1 = []
	svm_w2v_nostop_cvscores_precision = []
	svm_w2v_nostop_cvscores_recall = []
	
	svm_bert_cvscores_f1 = []
	svm_bert_cvscores_precision = []
	svm_bert_cvscores_recall = []



	nb_bow_cvscores_f1 = []
	nb_bow_cvscores_precision = []
	nb_bow_cvscores_recall = []
	
	nb_tf_stop_cvscores_f1 = []
	nb_tf_stop_cvscores_precision = []
	nb_tf_stop_cvscores_recall = []
	
	nb_tf_nostop_cvscores_f1 = []
	nb_tf_nostop_cvscores_precision = []
	nb_tf_nostop_cvscores_recall = []
	
	nb_w2v_stop_cvscores_f1 = []
	nb_w2v_stop_cvscores_precision = []
	nb_w2v_stop_cvscores_recall = []
	
	nb_w2v_nostop_cvscores_f1 = []
	nb_w2v_nostop_cvscores_precision = []
	nb_w2v_nostop_cvscores_recall = []
	
	nb_bert_cvscores_f1 = []
	nb_bert_cvscores_precision = []
	nb_bert_cvscores_recall = []



	dt_bow_cvscores_f1 = []
	dt_bow_cvscores_precision = []
	dt_bow_cvscores_recall = []
	
	dt_tf_stop_cvscores_f1 = []
	dt_tf_stop_cvscores_precision = []
	dt_tf_stop_cvscores_recall = []
	
	dt_tf_nostop_cvscores_f1 = []
	dt_tf_nostop_cvscores_precision = []
	dt_tf_nostop_cvscores_recall = []
	
	dt_w2v_stop_cvscores_f1 = []
	dt_w2v_stop_cvscores_precision = []
	dt_w2v_stop_cvscores_recall = []
	
	dt_w2v_nostop_cvscores_f1 = []
	dt_w2v_nostop_cvscores_precision = []
	dt_w2v_nostop_cvscores_recall = []
	
	dt_bert_cvscores_f1 = []
	dt_bert_cvscores_precision = []
	dt_bert_cvscores_recall = []
	
	rf_bow_cvscores_f1 = []
	rf_bow_cvscores_precision = []
	rf_bow_cvscores_recall = []
	
	rf_tf_stop_cvscores_f1 = []
	rf_tf_stop_cvscores_precision = []
	rf_tf_stop_cvscores_recall = []
	
	rf_tf_nostop_cvscores_f1 = []
	rf_tf_nostop_cvscores_precision = []
	rf_tf_nostop_cvscores_recall = []
	
	rf_w2v_stop_cvscores_f1 = []
	rf_w2v_stop_cvscores_precision = []
	rf_w2v_stop_cvscores_recall = []
	
	rf_w2v_nostop_cvscores_f1 = []
	rf_w2v_nostop_cvscores_precision = []
	rf_w2v_nostop_cvscores_recall = []
	
	rf_bert_cvscores_f1 = []
	rf_bert_cvscores_precision = []
	rf_bert_cvscores_recall = []
	
	mlp_cvscores_f1 = []
	mlp_cvscores_precision = []
	mlp_cvscores_recall = []
	
	#split into train and test
	for train_index, test_index in kf.split(corpus, corpus['gold_label']):
	
		print('----- SPLIT ', split_count, ' -----')
		
		train_data, test_data = corpus.iloc[train_index], corpus.iloc[test_index]
		Y_train, Y_test = corpus['gold_label'].iloc[train_index], corpus['gold_label'].iloc[train_index]
		
		embedding = ''
		# for Word2Vec with stop words
		train_data['clean_text'] = train_data['sentence'].apply(cleanText)
		test_data['clean_text'] = test_data['sentence'].apply(cleanText)
		
		# for BOW and TFIDF(s)
		train_data['clean_text_normalized'] = train_data['sentence'].apply(NormalizeWithPOS)
		test_data['clean_text_normalized'] = test_data['sentence'].apply(NormalizeWithPOS)
		
		freq_train = pd.Series(' '.join(train_data['clean_text_normalized']).split()).value_counts()
		less_five_freq_train = freq_train[(freq_train <5)]
		print(less_five_freq_train)
		
		freq_test = pd.Series(' '.join(test_data['clean_text_normalized']).split()).value_counts()
		less_five_freq_test = freq_test[(freq_test <5)]
		print(less_five_freq_test)
		
		train_data['clean_text_normalized'] = train_data['clean_text_normalized'].apply(lambda x: ' '.join(x for x in x.split() if x not in less_five_freq_train))
		test_data['clean_text_normalized'] = test_data['clean_text_normalized'].apply(lambda x: ' '.join(x for x in x.split() if x not in less_five_freq_test))
		
		embedding = 'WORD2VEC_NO_STOP'
		# for Word2Vec without stop words
		train_data['clean_text_w2v_no_stop'] = train_data['sentence'].apply(cleanText)
		test_data['clean_text_w2v_no_stop'] = test_data['sentence'].apply(cleanText)
		
		embedding = 'BERT'
		# for BERT(s)
		train_data['clean_text_bert'] = train_data['sentence'].apply(cleanText)
		test_data['clean_text_bert'] = test_data['sentence'].apply(cleanText)
		
		# Debugging
		#print("A review example of dataset before cleaning:", end="\n\n")
		#print(train_data.iloc[3]['text'], end='\n\n')

		#print("clean_text:")
		#print(train_data.iloc[3]['clean_text'], end="\n\n")

		#print("clean_text_normalized:")
		#print(train_data.iloc[3]['clean_text_normalized'], end="\n\n")

		#print("clean_text_w2v_no_stop:")
		#print(train_data.iloc[3]['clean_text_w2v_no_stop'], end="\n\n")

		#print("clean_text_bert:")
		#print(train_data.iloc[3]['clean_text_bert'], end="\n\n")
		
		embedding = 'BOW'
		vectorizer = CountVectorizer(stop_words='english')

		bow_training_features = vectorizer.fit_transform(train_data["clean_text_normalized"])	 
		bow_test_features = vectorizer.transform(test_data["clean_text_normalized"])

		print(bow_training_features.shape)
		
		embedding = 'TFIDF_WITH_STOP'
		vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)

		tfidf_with_stop_training_features = vectorizer.fit_transform(train_data["clean_text_normalized"])	 
		tfidf_with_stop_test_features = vectorizer.transform(test_data["clean_text_normalized"])

		print(tfidf_with_stop_training_features.shape)
		
		embedding = 'TFIDF_NO_STOP'
		vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=20000)

		tfidf_no_stop_training_features = vectorizer.fit_transform(train_data["clean_text_normalized"])	   
		tfidf_no_stop_test_features = vectorizer.transform(test_data["clean_text_normalized"])

		print(tfidf_no_stop_training_features.shape)
		
		embedding = 'WORD2VEC_WITH_STOP'
		words = train_data['clean_text'].apply(wordTokenize)
		w2v_with_stop_model = gensim.models.Word2Vec(words, min_count = 2, size = 100, window = 5) 
		word2vec_with_stop_training_features = train_data['clean_text'].apply(embeddToWord2Vec)
		word2vec_with_stop_test_features = test_data['clean_text'].apply(embeddToWord2Vec)
		feature = [x for x in word2vec_with_stop_training_features.transpose()]
		word2vec_with_stop_training_features = np.asarray(feature)

		feature = [x for x in word2vec_with_stop_test_features.transpose()]
		word2vec_with_stop_test_features = np.asarray(feature)

		print(word2vec_with_stop_training_features.shape)
		
		embedding = 'WORD2VEC_NO_STOP'
		words = train_data['clean_text_w2v_no_stop'].apply(wordTokenize)
		w2v_no_stop_model = gensim.models.Word2Vec(words, min_count = 2, size = 100, window = 5) 
		word2vec_no_stop_training_features = train_data['clean_text_w2v_no_stop'].apply(embeddToWord2Vec)
		word2vec_no_stop_test_features = test_data['clean_text_w2v_no_stop'].apply(embeddToWord2Vec)
		feature = [x for x in word2vec_no_stop_training_features.transpose()]
		word2vec_no_stop_training_features = np.asarray(feature)

		feature = [x for x in word2vec_no_stop_test_features.transpose()]
		word2vec_no_stop_test_features = np.asarray(feature)

		print(word2vec_no_stop_training_features.shape)
		
		# Cleaning before BERT
		embedding = 'BERT'
		 
		bert_transformers = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
		bert_version = 'SENTENCE'
		bert_sentence_training_features = train_data['clean_text_bert'].apply(embeddToBERT)
		bert_sentence_test_features = test_data['clean_text_bert'].apply(embeddToBERT)
		feature = [x for x in bert_sentence_training_features.transpose()]
		bert_sentence_training_features = np.asarray(feature)

		feature = [x for x in bert_sentence_test_features.transpose()]
		bert_sentence_test_features = np.asarray(feature)

		print(bert_sentence_training_features.shape)


		# START CV FOR ALL MODELS
		print('----------------------------------------- Training KNN classifier ........................................')
		# KNN classifier
		knn_param = {'n_neighbors': 3}
		# KNN classifier
		model = model = KNeighborsClassifier(**knn_param)

		# Training
		model.fit(bow_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_bow_knn = model.predict(bow_test_features)
		y_prob_bow_knn = model.predict_proba(bow_test_features)[:,1]

		# Result
		print("***** KNN BOW *****")
		knn_bow_f1, knn_bow_precision, knn_bow_recall = printResult(y_pred_bow_knn, y_prob_bow_knn,
																	'KNN_BOW' + str(split_count))
		print("\n")
		knn_bow_cvscores_f1.append(knn_bow_f1)
		knn_bow_cvscores_precision.append(knn_bow_precision)
		knn_bow_cvscores_recall.append(knn_bow_recall)

		# Training
		model.fit(tfidf_with_stop_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_tfidf_with_stop_knn = model.predict(tfidf_with_stop_test_features)
		y_prob_tfidf_with_stop_knn = model.predict_proba(tfidf_with_stop_test_features)[:,1]

		# Result
		print("***** KNN TF-IDF STOPWORDS *****")
		knn_tf_stop_f1, knn_tf_stop_precision, knn_tf_stop_recall = printResult(y_pred_tfidf_with_stop_knn,
																				y_prob_tfidf_with_stop_knn,
																				'KNN_TF-IDF_STOPWORDS' + str(
																					split_count))
		print("\n")
		knn_tf_stop_cvscores_f1.append(knn_tf_stop_f1)
		knn_tf_stop_cvscores_precision.append(knn_tf_stop_precision)
		knn_tf_stop_cvscores_recall.append(knn_tf_stop_recall)

		# Training
		model.fit(tfidf_no_stop_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_tfidf_no_stop_knn = model.predict(tfidf_no_stop_test_features)
		y_prob_tfidf_no_stop_knn = model.predict_proba(tfidf_no_stop_test_features)[:,1]

		# Result
		print("***** KNN TF-IDF NO STOPWORDS *****")
		knn_tf_nostop_f1, knn_tf_nostop_precision, knn_tf_nostop_recall = printResult(y_pred_tfidf_no_stop_knn,
																					  y_prob_tfidf_no_stop_knn,
																					  'KNN_TF-IDF_NOSTOPWORDS' + str(
																						  split_count))
		print("\n")
		knn_tf_nostop_cvscores_f1.append(knn_tf_nostop_f1)
		knn_tf_nostop_cvscores_precision.append(knn_tf_nostop_precision)
		knn_tf_nostop_cvscores_recall.append(knn_tf_nostop_recall)

		# Training
		model.fit(word2vec_with_stop_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_w2v_with_stop_knn = model.predict(word2vec_with_stop_test_features)
		y_prob_w2v_with_stop_knn = model.predict_proba(word2vec_with_stop_test_features)[:,1]

		# Result
		print("***** KNN W2VEC STOPWORDS *****")
		knn_w2v_stop_f1, knn_w2v_stop_precision, knn_w2v_stop_recall = printResult(y_pred_w2v_with_stop_knn,
																				   y_prob_w2v_with_stop_knn,
																				   'KNN_W2VEC_STOPWORDS' + str(
																					   split_count))
		print("\n")
		knn_w2v_stop_cvscores_f1.append(knn_w2v_stop_f1)
		knn_w2v_stop_cvscores_precision.append(knn_w2v_stop_precision)
		knn_w2v_stop_cvscores_recall.append(knn_w2v_stop_recall)

		# Training
		model.fit(word2vec_no_stop_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_w2v_no_stop_knn = model.predict(word2vec_no_stop_test_features)
		y_prob_w2v_no_stop_knn = model.predict_proba(word2vec_no_stop_test_features)[:,1]

		# Result
		print("***** KNN W2VEC NO STOPWORDS *****")
		knn_w2v_nostop_f1, knn_w2v_nostop_precision, knn_w2v_nostop_recall = printResult(y_pred_w2v_no_stop_knn,
																						 y_prob_w2v_no_stop_knn,
																						 'KNN_W2VEC_NOSTOPWORDS' + str(
																							 split_count))
		print("\n")
		knn_w2v_nostop_cvscores_f1.append(knn_w2v_nostop_f1)
		knn_w2v_nostop_cvscores_precision.append(knn_w2v_nostop_precision)
		knn_w2v_nostop_cvscores_recall.append(knn_w2v_nostop_recall)

		# Training
		model.fit(bert_sentence_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_bert_sentences_knn = model.predict(bert_sentence_test_features)
		y_prob_bert_sentences_knn = model.predict_proba(bert_sentence_test_features)[:,1]

		# Result
		print("***** KNN BERT SENTENCES *****")
		knn_bert_f1, knn_bert_precision, knn_bert_recall = printResult(y_pred_bert_sentences_knn,
																	   y_prob_bert_sentences_knn,
																	   'KNN_BERT_SENTENCES' + str(split_count))
		print("\n")
		knn_bert_cvscores_f1.append(knn_bert_f1)
		knn_bert_cvscores_precision.append(knn_bert_precision)
		knn_bert_cvscores_recall.append(knn_bert_recall)


		print('----------------------------------------- Training SVM classifier ........................................')

		# SVM classifier
		model = SVC(kernel ='linear', C = 1)
		
		# Training 
		model.fit(bow_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_bow_svm = model.predict(bow_test_features)
		y_prob_bow_svm = model.decision_function(bow_test_features)
		
		# Result
		print("***** SVM BOW *****")
		svm_bow_f1, svm_bow_precision, svm_bow_recall = printResult(y_pred_bow_svm, y_prob_bow_svm, 'SVM_BOW'+str(split_count))
		print("\n")
		svm_bow_cvscores_f1.append(svm_bow_f1)
		svm_bow_cvscores_precision.append(svm_bow_precision)
		svm_bow_cvscores_recall.append(svm_bow_recall)
		
		# Training 
		model.fit(tfidf_with_stop_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_tfidf_with_stop_svm = model.predict(tfidf_with_stop_test_features)
		y_prob_tfidf_with_stop_svm = model.decision_function(tfidf_with_stop_test_features)
		
		# Result
		print("***** SVM TF-IDF STOPWORDS *****")
		svm_tf_stop_f1, svm_tf_stop_precision, svm_tf_stop_recall = printResult(y_pred_tfidf_with_stop_svm, y_prob_tfidf_with_stop_svm, 'SVM_TF-IDF_STOPWORDS'+str(split_count))
		print("\n")
		svm_tf_stop_cvscores_f1.append(svm_tf_stop_f1)
		svm_tf_stop_cvscores_precision.append(svm_tf_stop_precision)
		svm_tf_stop_cvscores_recall.append(svm_tf_stop_recall)
		
		# Training 
		model.fit(tfidf_no_stop_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_tfidf_no_stop_svm = model.predict(tfidf_no_stop_test_features)
		y_prob_tfidf_no_stop_svm = model.decision_function(tfidf_no_stop_test_features)
		
		# Result
		print("***** SVM TF-IDF NO STOPWORDS *****")
		svm_tf_nostop_f1, svm_tf_nostop_precision, svm_tf_nostop_recall = printResult(y_pred_tfidf_no_stop_svm, y_prob_tfidf_no_stop_svm, 'SVM_TF-IDF_NOSTOPWORDS'+str(split_count))
		print("\n")
		svm_tf_nostop_cvscores_f1.append(svm_tf_nostop_f1)
		svm_tf_nostop_cvscores_precision.append(svm_tf_nostop_precision)
		svm_tf_nostop_cvscores_recall.append(svm_tf_nostop_recall)
		
		# Training 
		model.fit(word2vec_with_stop_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_w2v_with_stop_svm = model.predict(word2vec_with_stop_test_features)
		y_prob_w2v_with_stop_svm = model.decision_function(word2vec_with_stop_test_features)
		
		# Result
		print("***** SVM W2VEC STOPWORDS *****")
		svm_w2v_stop_f1, svm_w2v_stop_precision, svm_w2v_stop_recall = printResult(y_pred_w2v_with_stop_svm, y_prob_w2v_with_stop_svm, 'SVM_W2VEC_STOPWORDS'+str(split_count))
		print("\n")
		svm_w2v_stop_cvscores_f1.append(svm_w2v_stop_f1)
		svm_w2v_stop_cvscores_precision.append(svm_w2v_stop_precision)
		svm_w2v_stop_cvscores_recall.append(svm_w2v_stop_recall)
		
		# Training 
		model.fit(word2vec_no_stop_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_w2v_no_stop_svm = model.predict(word2vec_no_stop_test_features)
		y_prob_w2v_no_stop_svm = model.decision_function(word2vec_no_stop_test_features)
		
		# Result
		print("***** SVM W2VEC NO STOPWORDS *****")
		svm_w2v_nostop_f1, svm_w2v_nostop_precision, svm_w2v_nostop_recall = printResult(y_pred_w2v_no_stop_svm, y_prob_w2v_no_stop_svm, 'SVM_W2VEC_NOSTOPWORDS'+str(split_count))
		print("\n")
		svm_w2v_nostop_cvscores_f1.append(svm_w2v_nostop_f1)
		svm_w2v_nostop_cvscores_precision.append(svm_w2v_nostop_precision)
		svm_w2v_nostop_cvscores_recall.append(svm_w2v_nostop_recall)
		
		# Training 
		model.fit(bert_sentence_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_bert_sentences_svm = model.predict(bert_sentence_test_features)
		y_prob_bert_sentences_svm = model.decision_function(bert_sentence_test_features)
		
		# Result
		print("***** SVM BERT SENTENCES *****")
		svm_bert_f1, svm_bert_precision, svm_bert_recall = printResult(y_pred_bert_sentences_svm, y_prob_bert_sentences_svm, 'SVM_BERT_SENTENCES'+str(split_count))
		print("\n")
		svm_bert_cvscores_f1.append(svm_bert_f1)
		svm_bert_cvscores_precision.append(svm_bert_precision)
		svm_bert_cvscores_recall.append(svm_bert_recall)
		
		# Naive Bayes classifier
		model = MultinomialNB()
		
		# Training
		model.fit(bow_training_features.toarray(), train_data["gold_label"])

		# Evaluation
		y_pred_bow_nb = model.predict(bow_test_features.toarray())

		y_prob_bow_nb = model.predict_proba(bow_test_features.toarray())[:,1]
		
		# Result
		print("***** NB BOW *****")
		nb_bow_f1, nb_bow_precision, nb_bow_recall = printResult(y_pred_bow_nb, y_prob_bow_nb, 'NB_BOW'+str(split_count))
		print("\n")
		nb_bow_cvscores_f1.append(nb_bow_f1)
		nb_bow_cvscores_precision.append(nb_bow_precision)
		nb_bow_cvscores_recall.append(nb_bow_recall)
		
		model = MultinomialNB()
		# Training
		model.fit(tfidf_with_stop_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_tfidf_with_stop_nb = model.predict(tfidf_with_stop_test_features)

		y_prob_tfidf_with_stop_nb = model.predict_proba(tfidf_with_stop_test_features)[:,1]
		
		# Result
		print("***** NB TF-IDF STOPWORDS *****")
		nb_tf_stop_f1, nb_tf_stop_precision, nb_tf_stop_recall = printResult(y_pred_tfidf_with_stop_nb, y_prob_tfidf_with_stop_nb, 'NB_TF-IDF_STOPWORDS'+str(split_count))
		print("\n")
		nb_tf_stop_cvscores_f1.append(nb_tf_stop_f1)
		nb_tf_stop_cvscores_precision.append(nb_tf_stop_precision)
		nb_tf_stop_cvscores_recall.append(nb_tf_stop_recall)
		
		model = MultinomialNB()
		# Training
		model.fit(tfidf_no_stop_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_tfidf_no_stop_nb = model.predict(tfidf_no_stop_test_features)

		y_prob_tfidf_no_stop_nb = model.predict_proba(tfidf_no_stop_test_features)[:,1]
		
		# Result
		print("***** NB TF-IDF NO STOPWORDS *****")
		nb_tf_nostop_f1, nb_tf_nostop_precision, nb_tf_nostop_recall = printResult(y_pred_tfidf_no_stop_nb, y_prob_tfidf_no_stop_nb, 'NB_TF-IDF_NOSTOPWORDS'+str(split_count))
		print("\n")
		nb_tf_nostop_cvscores_f1.append(nb_tf_nostop_f1)
		nb_tf_nostop_cvscores_precision.append(nb_tf_nostop_precision)
		nb_tf_nostop_cvscores_recall.append(nb_tf_nostop_recall)
		
		model = GaussianNB()
		# Training 
		model.fit(word2vec_with_stop_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_w2v_with_stop_nb = model.predict(word2vec_with_stop_test_features)

		y_prob_w2v_with_stop_nb = model.predict_proba(word2vec_with_stop_test_features)[:,1]
		
		# Result
		print("***** NB W2VEC STOPWORDS *****")
		nb_w2v_stop_f1, nb_w2v_stop_precision, nb_w2v_stop_recall = printResult(y_pred_w2v_with_stop_nb, y_prob_w2v_with_stop_nb, 'NB_W2VEC_STOPWORDS'+str(split_count))
		print("\n")
		nb_w2v_stop_cvscores_f1.append(nb_w2v_stop_f1)
		nb_w2v_stop_cvscores_precision.append(nb_w2v_stop_precision)
		nb_w2v_stop_cvscores_recall.append(nb_w2v_stop_recall)
		
		model = GaussianNB()
		# Training 
		model.fit(word2vec_no_stop_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_w2v_no_stop_nb = model.predict(word2vec_no_stop_test_features)

		y_prob_w2v_no_stop_nb = model.predict_proba(word2vec_no_stop_test_features)[:,1]
		
		# Result
		print("***** NB W2VEC NO STOPWORDS *****")
		nb_w2v_nostop_f1, nb_w2v_nostop_precision, nb_w2v_nostop_recall = printResult(y_pred_w2v_no_stop_nb, y_prob_w2v_no_stop_nb, 'NB_W2VEC_NOSTOPWORDS'+str(split_count))
		print("\n")
		nb_w2v_nostop_cvscores_f1.append(nb_w2v_nostop_f1)
		nb_w2v_nostop_cvscores_precision.append(nb_w2v_nostop_precision)
		nb_w2v_nostop_cvscores_recall.append(nb_w2v_nostop_recall)
		
		model = GaussianNB()

		# Training 
		model.fit(bert_sentence_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_bert_sentences_nb = model.predict(bert_sentence_test_features)

		y_prob_bert_sentences_nb = model.predict_proba(bert_sentence_test_features)[:,1]
		
		# Result
		print("***** NB BERT SENTENCES *****")
		nb_bert_f1, nb_bert_precision, nb_bert_recall = printResult(y_pred_bert_sentences_nb, y_prob_bert_sentences_nb, 'NB_BERT_SENTENCES'+str(split_count))
		print("\n")
		nb_bert_cvscores_f1.append(nb_bert_f1)
		nb_bert_cvscores_precision.append(nb_bert_precision)
		nb_bert_cvscores_recall.append(nb_bert_recall)
		
		# Decision Tree classifier
		model = DecisionTreeClassifier()
		
		# Training
		model.fit(bow_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_bow_dt = model.predict(bow_test_features)

		y_prob_bow_dt = model.predict_proba(bow_test_features)[:,1]
		
		# Result
		print("***** DT BOW *****")
		dt_bow_f1, dt_bow_precision, dt_bow_recall = printResult(y_pred_bow_dt, y_prob_bow_dt, 'DT_BOW'+str(split_count))
		print("\n")
		dt_bow_cvscores_f1.append(dt_bow_f1)
		dt_bow_cvscores_precision.append(dt_bow_precision)
		dt_bow_cvscores_recall.append(dt_bow_recall)
		
		# Training
		model.fit(tfidf_with_stop_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_tfidf_with_stop_dt = model.predict(tfidf_with_stop_test_features)

		y_prob_tfidf_with_stop_dt = model.predict_proba(tfidf_with_stop_test_features)[:,1]
		
		# Result
		print("***** DT TF-IDF STOPWORDS *****")
		dt_tf_stop_f1, dt_tf_stop_precision, dt_tf_stop_recall = printResult(y_pred_tfidf_with_stop_dt, y_prob_tfidf_with_stop_dt, 'DT_TF-IDF_STOPWORDS'+str(split_count))
		print("\n")
		dt_tf_stop_cvscores_f1.append(dt_tf_stop_f1)
		dt_tf_stop_cvscores_precision.append(dt_tf_stop_precision)
		dt_tf_stop_cvscores_recall.append(dt_tf_stop_recall)
		
		# Training
		model.fit(tfidf_no_stop_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_tfidf_no_stop_dt = model.predict(tfidf_no_stop_test_features)

		y_prob_tfidf_no_stop_dt = model.predict_proba(tfidf_no_stop_test_features)[:,1]
		
		# Result
		print("***** DT TF-IDF NO STOPWORDS *****")
		dt_tf_nostop_f1, dt_tf_nostop_precision, dt_tf_nostop_recall = printResult(y_pred_tfidf_no_stop_dt, y_prob_tfidf_no_stop_dt, 'DT_TF-IDF_NOSTOPWORDS'+str(split_count))
		print("\n")
		dt_tf_nostop_cvscores_f1.append(dt_tf_nostop_f1)
		dt_tf_nostop_cvscores_precision.append(dt_tf_nostop_precision)
		dt_tf_nostop_cvscores_recall.append(dt_tf_nostop_recall)
		
		# Training
		model.fit(word2vec_with_stop_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_w2v_with_stop_dt = model.predict(word2vec_with_stop_test_features)

		y_prob_w2v_with_stop_dt = model.predict_proba(word2vec_with_stop_test_features)[:,1]
		
		# Result
		print("***** DT W2VEC STOPWORDS *****")
		dt_w2v_stop_f1, dt_w2v_stop_precision, dt_w2v_stop_recall = printResult(y_pred_w2v_with_stop_dt, y_prob_w2v_with_stop_dt, 'DT_W2VEC_STOPWORDS'+str(split_count))
		print("\n")
		dt_w2v_stop_cvscores_f1.append(dt_w2v_stop_f1)
		dt_w2v_stop_cvscores_precision.append(dt_w2v_stop_precision)
		dt_w2v_stop_cvscores_recall.append(dt_w2v_stop_recall)
		
		# Training
		model.fit(word2vec_no_stop_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_w2v_no_stop_dt = model.predict(word2vec_no_stop_test_features)

		y_prob_w2v_no_stop_dt = model.predict_proba(word2vec_no_stop_test_features)[:,1]
		
		# Result
		print("***** DT W2VEC NO STOPWORDS *****")
		dt_w2v_nostop_f1, dt_w2v_nostop_precision, dt_w2v_nostop_recall = printResult(y_pred_w2v_no_stop_dt, y_prob_w2v_no_stop_dt, 'DT_W2VEC_NOSTOPWORDS'+str(split_count))
		print("\n")
		dt_w2v_nostop_cvscores_f1.append(dt_w2v_nostop_f1)
		dt_w2v_nostop_cvscores_precision.append(dt_w2v_nostop_precision)
		dt_w2v_nostop_cvscores_recall.append(dt_w2v_nostop_recall)
		
		# Training
		model.fit(bert_sentence_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_bert_sentences_dt = model.predict(bert_sentence_test_features)

		y_prob_bert_sentences_dt = model.predict_proba(bert_sentence_test_features)[:,1]
		
		# Result
		print("***** DT BERT SENTENCES *****")
		dt_bert_f1, dt_bert_precision, dt_bert_recall = printResult(y_pred_bert_sentences_dt, y_prob_bert_sentences_dt, 'DT_BERT_SENTENCES'+str(split_count))
		print("\n")
		dt_bert_cvscores_f1.append(dt_bert_f1)
		dt_bert_cvscores_precision.append(dt_bert_precision)
		dt_bert_cvscores_recall.append(dt_bert_recall)
		
		# Random Forest classifier
		model = RandomForestClassifier(n_estimators=100, random_state=0)
		
		# Training
		model.fit(bow_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_bow_rf = model.predict(bow_test_features)

		y_prob_bow_rf = model.predict_proba(bow_test_features)[:,1]
		
		# Result
		print("***** RF BOW *****")
		rf_bow_f1, rf_bow_precision, rf_bow_recall = printResult(y_pred_bow_rf, y_prob_bow_rf, 'RF_BOW'+str(split_count))
		print("\n")
		rf_bow_cvscores_f1.append(rf_bow_f1)
		rf_bow_cvscores_precision.append(rf_bow_precision)
		rf_bow_cvscores_recall.append(rf_bow_recall)
		
		# Training
		model.fit(tfidf_with_stop_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_tfidf_with_stop_rf = model.predict(tfidf_with_stop_test_features)

		y_prob_tfidf_with_stop_rf = model.predict_proba(tfidf_with_stop_test_features)[:,1]
		
		# Result
		print("***** RF TF-IDF STOPWORDS *****")
		rf_tf_stop_f1, rf_tf_stop_precision, rf_tf_stop_recall = printResult(y_pred_tfidf_with_stop_rf, y_prob_tfidf_with_stop_rf, 'RF_TF-IDF_STOPWORDS'+str(split_count))
		print("\n")
		rf_tf_stop_cvscores_f1.append(rf_tf_stop_f1)
		rf_tf_stop_cvscores_precision.append(rf_tf_stop_precision)
		rf_tf_stop_cvscores_recall.append(rf_tf_stop_recall)
		
		# Training
		model.fit(tfidf_no_stop_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_tfidf_no_stop_rf = model.predict(tfidf_no_stop_test_features)

		y_prob_tfidf_no_stop_rf = model.predict_proba(tfidf_no_stop_test_features)[:,1]
		
		# Result
		print("***** RF TF-IDF NO STOPWORDS *****")
		rf_tf_nostop_f1, rf_tf_nostop_precision, rf_tf_nostop_recall = printResult(y_pred_tfidf_no_stop_rf, y_prob_tfidf_no_stop_rf, 'RF_TF-IDF_NOSTOPWORDS'+str(split_count))
		print("\n")
		rf_tf_nostop_cvscores_f1.append(rf_tf_nostop_f1)
		rf_tf_nostop_cvscores_precision.append(rf_tf_nostop_precision)
		rf_tf_nostop_cvscores_recall.append(rf_tf_nostop_recall)
		
		# Training
		model.fit(word2vec_with_stop_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_w2v_with_stop_rf = model.predict(word2vec_with_stop_test_features)

		y_prob_w2v_with_stop_rf = model.predict_proba(word2vec_with_stop_test_features)[:,1]
		
		# Result
		print("***** RF W2VEC STOPWORDS *****")
		rf_w2v_stop_f1, rf_w2v_stop_precision, rf_w2v_stop_recall = printResult(y_pred_w2v_with_stop_rf, y_prob_w2v_with_stop_rf, 'RF_W2VEC_STOPWORDS'+str(split_count))
		print("\n")
		rf_w2v_stop_cvscores_f1.append(rf_w2v_stop_f1)
		rf_w2v_stop_cvscores_precision.append(rf_w2v_stop_precision)
		rf_w2v_stop_cvscores_recall.append(rf_w2v_stop_recall)
		
		# Training
		model.fit(word2vec_no_stop_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_w2v_no_stop_rf = model.predict(word2vec_no_stop_test_features)

		y_prob_w2v_no_stop_rf = model.predict_proba(word2vec_no_stop_test_features)[:,1]
		
		# Result
		print("***** RF W2VEC NO STOPWORDS *****")
		rf_w2v_nostop_f1, rf_w2v_nostop_precision, rf_w2v_nostop_recall = printResult(y_pred_w2v_no_stop_rf, y_prob_w2v_no_stop_rf, 'RF_W2VEC_NOSTOPWORDS'+str(split_count))
		print("\n")
		rf_w2v_nostop_cvscores_f1.append(rf_w2v_nostop_f1)
		rf_w2v_nostop_cvscores_precision.append(rf_w2v_nostop_precision)
		rf_w2v_nostop_cvscores_recall.append(rf_w2v_nostop_recall)
		
		# Training
		model.fit(bert_sentence_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_bert_sentences_rf = model.predict(bert_sentence_test_features)

		y_prob_bert_sentences_rf = model.predict_proba(bert_sentence_test_features)[:,1]
		
		# Result
		print("***** RF BERT SENTENCES *****")
		rf_bert_f1, rf_bert_precision, rf_bert_recall = printResult(y_pred_bert_sentences_rf, y_prob_bert_sentences_rf, 'RF_BERT_SENTENCES'+str(split_count))
		print("\n")
		rf_bert_cvscores_f1.append(rf_bert_f1)
		rf_bert_cvscores_precision.append(rf_bert_precision)
		rf_bert_cvscores_recall.append(rf_bert_recall)
		
		# MLP classifier
		model = MLPClassifier(solver='adam', alpha=2e-4, hidden_layer_sizes=(5, 2), max_iter=400)
		
		# Training
		model.fit(bert_sentence_training_features, train_data["gold_label"])

		# Evaluation
		y_pred_bert_sentences_mlp = model.predict(bert_sentence_test_features)
		y_prob_bert_sentences_mlp = model.predict_proba(bert_sentence_test_features)[:,1]
		
		# Result
		print("***** MLP BERT SENTENCES *****")
		mlp_f1, mlp_precision, mlp_recall = printResult(y_pred_bert_sentences_mlp, y_prob_bert_sentences_mlp, 'MLP_BERT_SENTENCES'+str(split_count))
		print("\n")
		mlp_cvscores_f1.append(mlp_f1)
		mlp_cvscores_precision.append(mlp_precision)
		mlp_cvscores_recall.append(mlp_recall)
		
		split_count+= 1
	
	print("***** SVM BOW cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(svm_bow_cvscores_f1), np.std(svm_bow_cvscores_f1)))
	print("***** SVM BOW cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(svm_bow_cvscores_precision), np.std(svm_bow_cvscores_precision)))
	print("***** SVM BOW cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(svm_bow_cvscores_recall), np.std(svm_bow_cvscores_recall)))
	print("\n")
	print("***** SVM TF-IDF STOPWORDS cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(svm_tf_stop_cvscores_f1), np.std(svm_tf_stop_cvscores_f1)))
	print("***** SVM TF-IDF STOPWORDS cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(svm_tf_stop_cvscores_precision), np.std(svm_tf_stop_cvscores_precision)))
	print("***** SVM TF-IDF STOPWORDS cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(svm_tf_stop_cvscores_recall), np.std(svm_tf_stop_cvscores_recall)))
	print("\n")
	print("***** SVM TF-IDF NO STOPWORDS cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(svm_tf_nostop_cvscores_f1), np.std(svm_tf_nostop_cvscores_f1)))
	print("***** SVM TF-IDF NO STOPWORDS cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(svm_tf_nostop_cvscores_precision), np.std(svm_tf_nostop_cvscores_precision)))
	print("***** SVM TF-IDF NO STOPWORDS cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(svm_tf_nostop_cvscores_recall), np.std(svm_tf_nostop_cvscores_recall)))
	print("\n")
	print("***** SVM W2V STOPWORDS cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(svm_w2v_stop_cvscores_f1), np.std(svm_w2v_stop_cvscores_f1)))
	print("***** SVM W2V STOPWORDS cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(svm_w2v_stop_cvscores_precision), np.std(svm_w2v_stop_cvscores_precision)))
	print("***** SVM W2V STOPWORDS cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(svm_w2v_stop_cvscores_recall), np.std(svm_w2v_stop_cvscores_recall)))
	print("\n")
	print("***** SVM W2V NO STOPWORDS cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(svm_w2v_nostop_cvscores_f1), np.std(svm_w2v_nostop_cvscores_f1)))
	print("***** SVM W2V NO STOPWORDS cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(svm_w2v_nostop_cvscores_precision), np.std(svm_w2v_nostop_cvscores_precision)))
	print("***** SVM W2V NO STOPWORDS cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(svm_w2v_nostop_cvscores_recall), np.std(svm_w2v_nostop_cvscores_recall)))
	print("\n")
	print("***** SVM BERT cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(svm_bert_cvscores_f1), np.std(svm_bert_cvscores_f1)))
	print("***** SVM BERT cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(svm_bert_cvscores_precision), np.std(svm_bert_cvscores_precision)))
	print("***** SVM BERT cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(svm_bert_cvscores_recall), np.std(svm_bert_cvscores_recall)))
	print("\n")
	print("***** NB BOW cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(nb_bow_cvscores_f1), np.std(nb_bow_cvscores_f1)))
	print("***** NB BOW cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(nb_bow_cvscores_precision), np.std(nb_bow_cvscores_precision)))
	print("***** NB BOW cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(nb_bow_cvscores_recall), np.std(nb_bow_cvscores_recall)))
	print("\n")
	print("***** NB TF-IDF STOPWORDS cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(nb_tf_stop_cvscores_f1), np.std(nb_tf_stop_cvscores_f1)))
	print("***** NB TF-IDF STOPWORDS cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(nb_tf_stop_cvscores_precision), np.std(nb_tf_stop_cvscores_precision)))
	print("***** NB TF-IDF STOPWORDS cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(nb_tf_stop_cvscores_recall), np.std(nb_tf_stop_cvscores_recall)))
	print("\n")
	print("***** NB TF-IDF NO STOPWORDS cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(nb_tf_nostop_cvscores_f1), np.std(nb_tf_nostop_cvscores_f1)))
	print("***** NB TF-IDF NO STOPWORDS cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(nb_tf_nostop_cvscores_precision), np.std(nb_tf_nostop_cvscores_precision)))
	print("***** NB TF-IDF NO STOPWORDS cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(nb_tf_nostop_cvscores_recall), np.std(nb_tf_nostop_cvscores_recall)))
	print("\n")
	print("***** NB W2V STOPWORDS cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(nb_w2v_stop_cvscores_f1), np.std(nb_w2v_stop_cvscores_f1)))
	print("***** NB W2V STOPWORDS cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(nb_w2v_stop_cvscores_precision), np.std(nb_w2v_stop_cvscores_precision)))
	print("***** NB W2V STOPWORDS cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(nb_w2v_stop_cvscores_recall), np.std(nb_w2v_stop_cvscores_recall)))
	print("\n")
	print("***** NB W2V NO STOPWORDS cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(nb_w2v_nostop_cvscores_f1), np.std(nb_w2v_nostop_cvscores_f1)))
	print("***** NB W2V NO STOPWORDS cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(nb_w2v_nostop_cvscores_precision), np.std(nb_w2v_nostop_cvscores_precision)))
	print("***** NB W2V NO STOPWORDS cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(nb_w2v_nostop_cvscores_recall), np.std(nb_w2v_nostop_cvscores_recall)))
	print("\n")
	print("***** NB BERT cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(nb_bert_cvscores_f1), np.std(nb_bert_cvscores_f1)))
	print("***** NB BERT cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(nb_bert_cvscores_precision), np.std(nb_bert_cvscores_precision)))
	print("***** NB BERT cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(nb_bert_cvscores_recall), np.std(nb_bert_cvscores_recall)))
	print("\n")
	print("***** DT BOW cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(dt_bow_cvscores_f1), np.std(dt_bow_cvscores_f1)))
	print("***** DT BOW cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(dt_bow_cvscores_precision), np.std(dt_bow_cvscores_precision)))
	print("***** DT BOW cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(dt_bow_cvscores_recall), np.std(dt_bow_cvscores_recall)))
	print("\n")
	print("***** DT TF-IDF STOPWORDS cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(dt_tf_stop_cvscores_f1), np.std(dt_tf_stop_cvscores_f1)))
	print("***** DT TF-IDF STOPWORDS cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(dt_tf_stop_cvscores_precision), np.std(dt_tf_stop_cvscores_precision)))
	print("***** DT TF-IDF STOPWORDS cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(dt_tf_stop_cvscores_recall), np.std(dt_tf_stop_cvscores_recall)))
	print("\n")
	print("***** DT TF-IDF NO STOPWORDS cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(dt_tf_nostop_cvscores_f1), np.std(dt_tf_nostop_cvscores_f1)))
	print("***** DT TF-IDF NO STOPWORDS cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(dt_tf_nostop_cvscores_precision), np.std(dt_tf_nostop_cvscores_precision)))
	print("***** DT TF-IDF NO STOPWORDS cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(dt_tf_nostop_cvscores_recall), np.std(dt_tf_nostop_cvscores_recall)))
	print("\n")
	print("***** DT W2V STOPWORDS cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(dt_w2v_stop_cvscores_f1), np.std(dt_w2v_stop_cvscores_f1)))
	print("***** DT W2V STOPWORDS cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(dt_w2v_stop_cvscores_precision), np.std(dt_w2v_stop_cvscores_precision)))
	print("***** DT W2V STOPWORDS cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(dt_w2v_stop_cvscores_recall), np.std(dt_w2v_stop_cvscores_recall)))
	print("\n")
	print("***** DT W2V NO STOPWORDS cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(dt_w2v_nostop_cvscores_f1), np.std(dt_w2v_nostop_cvscores_f1)))
	print("***** DT W2V NO STOPWORDS cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(dt_w2v_nostop_cvscores_precision), np.std(dt_w2v_nostop_cvscores_precision)))
	print("***** DT W2V NO STOPWORDS cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(dt_w2v_nostop_cvscores_recall), np.std(dt_w2v_nostop_cvscores_recall)))
	print("\n")
	print("***** DT BERT cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(dt_bert_cvscores_f1), np.std(dt_bert_cvscores_f1)))
	print("***** DT BERT cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(dt_bert_cvscores_precision), np.std(dt_bert_cvscores_precision)))
	print("***** DT BERT cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(dt_bert_cvscores_recall), np.std(dt_bert_cvscores_recall)))
	print("\n")
	print("***** RF BOW cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(rf_bow_cvscores_f1), np.std(rf_bow_cvscores_f1)))
	print("***** RF BOW cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(rf_bow_cvscores_precision), np.std(rf_bow_cvscores_precision)))
	print("***** RF BOW cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(rf_bow_cvscores_recall), np.std(rf_bow_cvscores_recall)))
	print("\n")
	print("***** RF TF-IDF STOPWORDS cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(rf_tf_stop_cvscores_f1), np.std(rf_tf_stop_cvscores_f1)))
	print("***** RF TF-IDF STOPWORDS cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(rf_tf_stop_cvscores_precision), np.std(rf_tf_stop_cvscores_precision)))
	print("***** RF TF-IDF STOPWORDS cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(rf_tf_stop_cvscores_recall), np.std(rf_tf_stop_cvscores_recall)))
	print("\n")
	print("***** RF TF-IDF NO STOPWORDS cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(rf_tf_nostop_cvscores_f1), np.std(rf_tf_nostop_cvscores_f1)))
	print("***** RF TF-IDF NO STOPWORDS cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(rf_tf_nostop_cvscores_precision), np.std(rf_tf_nostop_cvscores_precision)))
	print("***** RF TF-IDF NO STOPWORDS cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(rf_tf_nostop_cvscores_recall), np.std(rf_tf_nostop_cvscores_recall)))
	print("\n")
	print("***** RF W2V STOPWORDS cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(rf_w2v_stop_cvscores_f1), np.std(rf_w2v_stop_cvscores_f1)))
	print("***** RF W2V STOPWORDS cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(rf_w2v_stop_cvscores_precision), np.std(rf_w2v_stop_cvscores_precision)))
	print("***** RF W2V STOPWORDS cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(rf_w2v_stop_cvscores_recall), np.std(rf_w2v_stop_cvscores_recall)))
	print("\n")
	print("***** RF W2V NO STOPWORDS cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(rf_w2v_nostop_cvscores_f1), np.std(rf_w2v_nostop_cvscores_f1)))
	print("***** RF W2V NO STOPWORDS cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(rf_w2v_nostop_cvscores_precision), np.std(rf_w2v_nostop_cvscores_precision)))
	print("***** RF W2V NO STOPWORDS cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(rf_w2v_nostop_cvscores_recall), np.std(rf_w2v_nostop_cvscores_recall)))
	print("\n")
	print("***** RF BERT cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(rf_bert_f1), np.std(rf_bert_f1)))
	print("***** RF BERT cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(rf_bert_precision), np.std(rf_bert_precision)))
	print("***** RF BERT cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(rf_bert_recall), np.std(rf_bert_recall)))
	print("\n")
	print("***** MLP BERT cross validation f1 score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(mlp_cvscores_f1), np.std(mlp_cvscores_f1)))
	print("***** MLP BERT cross validation precision score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(mlp_cvscores_precision), np.std(mlp_cvscores_precision)))
	print("***** MLP BERT cross validation recall score *****")
	print("%.2f%% (+/- %.2f%%)" % (np.mean(mlp_cvscores_recall), np.std(mlp_cvscores_recall)))
	print("\n")