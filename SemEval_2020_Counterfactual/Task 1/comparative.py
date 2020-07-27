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
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import gensim 
from gensim.models import Word2Vec
import sys


def save(filename, model):
	pickle.dump(model, open(filename, 'wb'))

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
	savepath =	'output/plots/comparative/'
	acc = accuracy_score(test_data["gold_label"], y_pred)
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
	plt.savefig(savepath+title+'.png')
	plt.close()
	
if __name__ == '__main__':
	sys.stdout = open('output/logs/log.txt', 'w') # redirect prints to log file
	path = "data/train.csv"
	corpus = pd.read_csv(path, encoding='utf-8')
	#split into train and test
	train_data, test_data = train_test_split(corpus, test_size=0.3)

	print('Init processing ...')
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

	print('Init training ...')
	print('---------------------------- Training KNN classifier ......................')
	# KNN classifier
	knn_param = {'n_neighbors': 3}
	model = KNeighborsClassifier(**knn_param)  # no n-estimators

	# Training
	model.fit(bow_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_bow_knn = model.predict(bow_test_features)
	y_prob_bow_knn = model.predict_proba(bow_test_features)[:,1]

	# Result
	print("***** KNN BOW *****")
	printResult(y_pred_bow_knn, y_prob_bow_knn, 'KNN_BOW')
	print("\n")

	# Training
	model.fit(tfidf_with_stop_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_tfidf_with_stop_knn = model.predict(tfidf_with_stop_test_features)
	y_prob_tfidf_with_stop_knn = model.predict_proba(tfidf_with_stop_test_features)[:,1]

	# Result
	print("***** KNN TF-IDF STOPWORDS *****")
	printResult(y_pred_tfidf_with_stop_knn, y_prob_tfidf_with_stop_knn, 'KNN_TF-IDF_STOPWORDS')
	print("\n")

	# Training
	model.fit(tfidf_no_stop_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_tfidf_no_stop_knn = model.predict(tfidf_no_stop_test_features)
	y_prob_tfidf_no_stop_knn = model.predict_proba(tfidf_no_stop_test_features)[:,1]

	# Result
	print("***** KNN TF-IDF NO STOPWORDS *****")
	printResult(y_pred_tfidf_no_stop_knn, y_prob_tfidf_no_stop_knn, 'KNN_TF-IDF_NOSTOPWORDS')
	print("\n")

	# Training
	model.fit(word2vec_with_stop_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_w2v_with_stop_knn = model.predict(word2vec_with_stop_test_features)
	y_prob_w2v_with_stop_knn = model.predict_proba(word2vec_with_stop_test_features)[:,1]

	# Result
	print("***** knn W2VEC STOPWORDS *****")
	printResult(y_pred_w2v_with_stop_knn, y_prob_w2v_with_stop_knn, 'KNN_W2VEC_STOPWORDS')
	print("\n")

	# Training
	model.fit(word2vec_no_stop_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_w2v_no_stop_knn = model.predict(word2vec_no_stop_test_features)
	y_prob_w2v_no_stop_knn = model.predict_proba(word2vec_with_stop_test_features)[:,1]

	# Result
	print("***** KNN W2VEC NO STOPWORDS *****")
	printResult(y_pred_w2v_no_stop_knn, y_prob_w2v_no_stop_knn, 'KNN_W2VEC_NOSTOPWORDS')
	print("\n")

	# Training
	model.fit(bert_sentence_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_bert_sentences_knn = model.predict(bert_sentence_test_features)
	y_prob_bert_sentences_knn = model.predict_proba(bert_sentence_test_features)[:,1]

	# Result
	print("***** KNN BERT SENTENCES *****")
	printResult(y_pred_bert_sentences_knn, y_prob_bert_sentences_knn, 'KNN_BERT_SENTENCES')
	print("\n")

	# SVM classifier
	model = SVC(kernel ='linear', C = 1)

	# Training
	model.fit(bow_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_bow_svm = model.predict(bow_test_features)
	y_prob_bow_svm = model.decision_function(bow_test_features)
	
	# Result
	print("***** SVM BOW *****")
	printResult(y_pred_bow_svm, y_prob_bow_svm, 'SVM_BOW')
	print("\n")
	
	# Training 
	model.fit(tfidf_with_stop_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_tfidf_with_stop_svm = model.predict(tfidf_with_stop_test_features)
	y_prob_tfidf_with_stop_svm = model.decision_function(tfidf_with_stop_test_features)
	
	# Result
	print("***** SVM TF-IDF STOPWORDS *****")
	printResult(y_pred_tfidf_with_stop_svm, y_prob_tfidf_with_stop_svm, 'SVM_TF-IDF_STOPWORDS')
	print("\n")
	
	# Training 
	model.fit(tfidf_no_stop_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_tfidf_no_stop_svm = model.predict(tfidf_no_stop_test_features)
	y_prob_tfidf_no_stop_svm = model.decision_function(tfidf_no_stop_test_features)
	
	# Result
	print("***** SVM TF-IDF NO STOPWORDS *****")
	printResult(y_pred_tfidf_no_stop_svm, y_prob_tfidf_no_stop_svm, 'SVM_TF-IDF_NOSTOPWORDS')
	print("\n")
	
	# Training 
	model.fit(word2vec_with_stop_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_w2v_with_stop_svm = model.predict(word2vec_with_stop_test_features)
	y_prob_w2v_with_stop_svm = model.decision_function(word2vec_with_stop_test_features)
	
	# Result
	print("***** SVM W2VEC STOPWORDS *****")
	printResult(y_pred_w2v_with_stop_svm, y_prob_w2v_with_stop_svm, 'SVM_W2VEC_STOPWORDS')
	print("\n")
	
	# Training 
	model.fit(word2vec_no_stop_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_w2v_no_stop_svm = model.predict(word2vec_no_stop_test_features)
	y_prob_w2v_no_stop_svm = model.decision_function(word2vec_no_stop_test_features)
	
	# Result
	print("***** SVM W2VEC NO STOPWORDS *****")
	printResult(y_pred_w2v_no_stop_svm, y_prob_w2v_no_stop_svm, 'SVM_W2VEC_NOSTOPWORDS')
	print("\n")
	
	# Training 
	model.fit(bert_sentence_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_bert_sentences_svm = model.predict(bert_sentence_test_features)
	y_prob_bert_sentences_svm = model.decision_function(bert_sentence_test_features)
	
	# Result
	print("***** SVM BERT SENTENCES *****")
	printResult(y_pred_bert_sentences_svm, y_prob_bert_sentences_svm, 'SVM_BERT_SENTENCES')
	print("\n")
	
	# Naive Bayes classifier
	model = MultinomialNB()
	
	# Training
	model.fit(bow_training_features.toarray(), train_data["gold_label"])

	# Evaluation
	y_pred_bow_nb = model.predict(bow_test_features.toarray())

	y_prob_bow_nb = model.predict_proba(bow_test_features.toarray())[:,1]
	
	# Result
	print("***** NB BOW *****")
	printResult(y_pred_bow_nb, y_prob_bow_nb, 'NB_BOW')
	print("\n")
	
	model = MultinomialNB()
	# Training
	model.fit(tfidf_with_stop_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_tfidf_with_stop_nb = model.predict(tfidf_with_stop_test_features)

	y_prob_tfidf_with_stop_nb = model.predict_proba(tfidf_with_stop_test_features)[:,1]
	
	# Result
	print("***** NB TF-IDF STOPWORDS *****")
	printResult(y_pred_tfidf_with_stop_nb, y_prob_tfidf_with_stop_nb, 'NB_TF-IDF_STOPWORDS')
	print("\n")
	
	model = MultinomialNB()
	# Training
	model.fit(tfidf_no_stop_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_tfidf_no_stop_nb = model.predict(tfidf_no_stop_test_features)

	y_prob_tfidf_no_stop_nb = model.predict_proba(tfidf_no_stop_test_features)[:,1]
	
	# Result
	print("***** NB TF-IDF NO STOPWORDS *****")
	printResult(y_pred_tfidf_no_stop_nb, y_prob_tfidf_no_stop_nb, 'NB_TF-IDF_NOSTOPWORDS')
	print("\n")
	
	model = GaussianNB()
	# Training 
	model.fit(word2vec_with_stop_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_w2v_with_stop_nb = model.predict(word2vec_with_stop_test_features)

	y_prob_w2v_with_stop_nb = model.predict_proba(word2vec_with_stop_test_features)[:,1]
	
	# Result
	print("***** NB W2VEC STOPWORDS *****")
	printResult(y_pred_w2v_with_stop_nb, y_prob_w2v_with_stop_nb, 'NB_W2VEC_STOPWORDS')
	print("\n")
	
	model = GaussianNB()
	# Training 
	model.fit(word2vec_no_stop_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_w2v_no_stop_nb = model.predict(word2vec_no_stop_test_features)

	y_prob_w2v_no_stop_nb = model.predict_proba(word2vec_no_stop_test_features)[:,1]
	
	# Result
	print("***** NB W2VEC NO STOPWORDS *****")
	printResult(y_pred_w2v_no_stop_nb, y_prob_w2v_no_stop_nb, 'NB_W2VEC_NOSTOPWORDS')
	print("\n")
	
	model = GaussianNB()

	# Training 
	model.fit(bert_sentence_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_bert_sentences_nb = model.predict(bert_sentence_test_features)

	y_prob_bert_sentences_nb = model.predict_proba(bert_sentence_test_features)[:,1]
	
	# Result
	print("***** NB BERT SENTENCES *****")
	printResult(y_pred_bert_sentences_nb, y_prob_bert_sentences_nb, 'NB_BERT_SENTENCES')
	print("\n")
	
	# Decision Tree classifier
	model = DecisionTreeClassifier()
	
	# Training
	model.fit(bow_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_bow_dt = model.predict(bow_test_features)

	y_prob_bow_dt = model.predict_proba(bow_test_features)[:,1]
	
	# Result
	print("***** DT BOW *****")
	printResult(y_pred_bow_dt, y_prob_bow_dt, 'DT_BOW')
	print("\n")
	
	# Training
	model.fit(tfidf_with_stop_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_tfidf_with_stop_dt = model.predict(tfidf_with_stop_test_features)

	y_prob_tfidf_with_stop_dt = model.predict_proba(tfidf_with_stop_test_features)[:,1]
	
	# Result
	print("***** DT TF-IDF STOPWORDS *****")
	printResult(y_pred_tfidf_with_stop_dt, y_prob_tfidf_with_stop_dt, 'DT_TF-IDF_STOPWORDS')
	print("\n")
	
	# Training
	model.fit(tfidf_no_stop_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_tfidf_no_stop_dt = model.predict(tfidf_no_stop_test_features)

	y_prob_tfidf_no_stop_dt = model.predict_proba(tfidf_no_stop_test_features)[:,1]
	
	# Result
	print("***** DT TF-IDF NO STOPWORDS *****")
	printResult(y_pred_tfidf_no_stop_dt, y_prob_tfidf_no_stop_dt, 'DT_TF-IDF_NOSTOPWORDS')
	print("\n")
	
	# Training
	model.fit(word2vec_with_stop_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_w2v_with_stop_dt = model.predict(word2vec_with_stop_test_features)

	y_prob_w2v_with_stop_dt = model.predict_proba(word2vec_with_stop_test_features)[:,1]
	
	# Result
	print("***** DT W2VEC STOPWORDS *****")
	printResult(y_pred_w2v_with_stop_dt, y_prob_w2v_with_stop_dt, 'DT_W2VEC_STOPWORDS')
	print("\n")
	
	# Training
	model.fit(word2vec_no_stop_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_w2v_no_stop_dt = model.predict(word2vec_no_stop_test_features)

	y_prob_w2v_no_stop_dt = model.predict_proba(word2vec_no_stop_test_features)[:,1]
	
	# Result
	print("***** DT W2VEC NO STOPWORDS *****")
	printResult(y_pred_w2v_no_stop_dt, y_prob_w2v_no_stop_dt, 'DT_W2VEC_NOSTOPWORDS')
	print("\n")
	
	# Training
	model.fit(bert_sentence_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_bert_sentences_dt = model.predict(bert_sentence_test_features)

	y_prob_bert_sentences_dt = model.predict_proba(bert_sentence_test_features)[:,1]
	
	# Result
	print("***** DT BERT SENTENCES *****")
	printResult(y_pred_bert_sentences_dt, y_prob_bert_sentences_dt, 'DT_BERT_SENTENCES')
	print("\n")
	
	# Random Forest classifier
	model = RandomForestClassifier(n_estimators=100, random_state=0)
	
	# Training
	model.fit(bow_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_bow_rf = model.predict(bow_test_features)

	y_prob_bow_rf = model.predict_proba(bow_test_features)[:,1]
	
	# Result
	print("***** RF BOW *****")
	printResult(y_pred_bow_rf, y_prob_bow_rf, 'RF_BOW')
	print("\n")
	
	# Training
	model.fit(tfidf_with_stop_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_tfidf_with_stop_rf = model.predict(tfidf_with_stop_test_features)

	y_prob_tfidf_with_stop_rf = model.predict_proba(tfidf_with_stop_test_features)[:,1]
	
	# Result
	print("***** RF TF-IDF STOPWORDS *****")
	printResult(y_pred_tfidf_with_stop_rf, y_prob_tfidf_with_stop_rf, 'RF_TF-IDF_STOPWORDS')
	print("\n")
	
	# Training
	model.fit(tfidf_no_stop_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_tfidf_no_stop_rf = model.predict(tfidf_no_stop_test_features)

	y_prob_tfidf_no_stop_rf = model.predict_proba(tfidf_no_stop_test_features)[:,1]
	
	# Result
	print("***** RF TF-IDF NO STOPWORDS *****")
	printResult(y_pred_tfidf_no_stop_rf, y_prob_tfidf_no_stop_rf, 'RF_TF-IDF_NOSTOPWORDS')
	print("\n")
	
	# Training
	model.fit(word2vec_with_stop_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_w2v_with_stop_rf = model.predict(word2vec_with_stop_test_features)

	y_prob_w2v_with_stop_rf = model.predict_proba(word2vec_with_stop_test_features)[:,1]
	
	# Result
	print("***** RF W2VEC STOPWORDS *****")
	printResult(y_pred_w2v_with_stop_rf, y_prob_w2v_with_stop_rf, 'RF_W2VEC_STOPWORDS')
	print("\n")
	
	# Training
	model.fit(word2vec_no_stop_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_w2v_no_stop_rf = model.predict(word2vec_no_stop_test_features)

	y_prob_w2v_no_stop_rf = model.predict_proba(word2vec_no_stop_test_features)[:,1]
	
	# Result
	print("***** RF W2VEC NO STOPWORDS *****")
	printResult(y_pred_w2v_no_stop_rf, y_prob_w2v_no_stop_rf, 'RF_W2VEC_NOSTOPWORDS')
	print("\n")
	
	# Training
	model.fit(bert_sentence_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_bert_sentences_rf = model.predict(bert_sentence_test_features)

	y_prob_bert_sentences_rf = model.predict_proba(bert_sentence_test_features)[:,1]
	
	# Result
	print("***** RF BERT SENTENCES *****")
	printResult(y_pred_bert_sentences_rf, y_prob_bert_sentences_rf, 'RF_BERT_SENTENCES')
	print("\n")
	
	# MLP classifier
	model = MLPClassifier(solver='adam', alpha=2e-4, hidden_layer_sizes=(5, 2), max_iter=400)
	
	# Training
	model.fit(bert_sentence_training_features, train_data["gold_label"])

	# Evaluation
	y_pred_bert_sentences_mlp = model.predict(bert_sentence_test_features)
	y_prob_bert_sentences_mlp = model.predict_proba(bert_sentence_test_features)[:,1]
	
	# Result
	print("***** MLP BERT SENTENCES *****")
	printResult(y_pred_bert_sentences_mlp, y_prob_bert_sentences_mlp, 'MLP_BERT_SENTENCES')
	print("\n")
	