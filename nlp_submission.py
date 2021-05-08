# !/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
#
# (c) Copyright University of Southampton, 2021
#
# Copyright in this software belongs to University of Southampton,
# Highfield, University Road, Southampton SO17 1BJ
#
# Created By : Stuart E. Middleton
# Created Date : 2021/01/29
# Project : Teaching
#
######################################################################

from __future__ import absolute_import, division, print_function, unicode_literals

import sys, codecs, json, math, time, warnings, re, logging
warnings.simplefilter( action='ignore', category=FutureWarning )

import nltk, numpy, scipy, sklearn, sklearn_crfsuite, sklearn_crfsuite.metrics

LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger( __name__ )
logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
logger.info('logging started')

def get_tagged_sentences(ontonotes:dict, max_sentences:int = 25000):
	tagged_sentences = []
	for sentences in ontonotes.values():
		for sentence in sentences.values():
			if len(tagged_sentences) >= max_sentences: return tagged_sentences
			if 'XX' in sentence.get('pos'):
				continue
			if 'VERB' in sentence.get('pos'):
				continue
			tokens = sentence.get('tokens')
			pos = sentence.get('pos')
			entities = sentence.get('ne')
			if entities and 'parse_error' not in entities.keys():
				tagged_sentence = []
				for token in range(len(tokens)):
					entity = None
					for ne in entities.values():
						if token in ne.get('tokens'):
							entity = ne
							break
					if entity is None:
						tagged_sentence.append((tokens[token], pos[token], 'O'))
					else:
						tokens_arr = entity.get('tokens')
						tag = entity.get('type')
						if len(tokens_arr) == 1:
							tagged_sentence.append((tokens[token], pos[token], tag + "-S"))
						elif token == tokens_arr[0]:
							tagged_sentence.append((tokens[token], pos[token], tag + "-B"))
						elif token == tokens_arr[-1]:
							tagged_sentence.append((tokens[token], pos[token], tag + "-E"))
						else:
							tagged_sentence.append((tokens[token], pos[token], tag + "-I"))
				if tagged_sentence: tagged_sentences.append(tagged_sentence)
	return tagged_sentences

def sent2features(sent):
	return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
	return [label for token, postag, label in sent]

def sent2tokens(sent):
	return [token for token, postag, label in sent]

def print_F1_scores( micro_F1 ) :
	for label in micro_F1 :
		logger.info( "%-15s -> f1 %0.2f ; prec %0.2f ; recall %0.2f" % ( label, micro_F1[label]['f1-score'], micro_F1[label]['precision'], micro_F1[label]['recall'] ) )

def print_transitions(trans_features):
	for (label_from, label_to), weight in trans_features:
		logger.info( "%-15s -> %-15s %0.6f" % (label_from, label_to, weight) )

def print_state_features(state_features):
	for (attr, label), weight in state_features:
		logger.info( "%0.6f %-15s %s" % (weight, label, attr) )

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features

def exec_ner( file_chapter = None, ontonotes_file = None ) :

	# INSERT CODE TO TRAIN A CRF NER MODEL TO TAG THE CHAPTER OF TEXT (subtask 3)
	display_label_subset = [ 'CARDINAL', 'ORDINAL', 'DATE', 'NORP', 'PERSON' ]
	# load parsed ontonotes dataset
	dataset = codecs.open(ontonotes_file, 'r', 'utf-8', errors = 'replace').read()
	ontonotes = json.loads(dataset)

	sentences = get_tagged_sentences(ontonotes)
	
	train, test = sklearn.model_selection.train_test_split(sentences, test_size=0.2)

	X_train = [sent2features(s) for s in train]
	y_train = [sent2labels(s) for s in train]

	X_test = [sent2features(s) for s in test]
	y_test = [sent2labels(s) for s in test]

	# get the label set
	set_labels = set([])
	for data in [y_train,y_test] :
		for n_sent in range(len(data)) :
			for str_label in data[n_sent] :
				set_labels.add( str_label )
	labels = list( set_labels )
	logger.info( '' )
	logger.info( 'labels = ' + repr(labels) )
	labels.remove('O')

	crf = sklearn_crfsuite.CRF(
		algorithm='lbfgs',
		c1=0.1,
		c2=0.1,
		max_iterations=20,
		all_possible_transitions=True,
	)

	crf.fit(X_train, y_train)
	# compute the macro F1 score (F1 for instances of each label class averaged) in the test set
	y_pred = crf.predict( X_test )
	sorted_labels = sorted(
		labels, 
		key=lambda name: (name[1:], name[0])
	)
	macro_scores = sklearn_crfsuite.metrics.flat_classification_report( y_test, y_pred, labels=sorted_labels, digits=3, output_dict = True )
	logger.info( '' )
	logger.info( 'macro F1 scores'  )
	print_F1_scores( macro_scores )
	print()
	

	# USING NER MODEL AND REGEX GENERATE A SET OF BOOK CHARACTERS AND FILTERED SET OF NE TAGS (subtask 4)

	# hardcoded output to show exactly what is expected to be serialized (you should change this)
	dictNE = {
			"CARDINAL": [
				"two",
				"three",
				"one"
			],
			"ORDINAL": [
				"first"
			],
			"DATE": [
				"saturday",
			],
			"NORP": [
				"indians"
			],
			"PERSON": [
				"creakle",
				"mr. creakle",
				"mrs. creakle",
				"miss creakle"
			]
		}

	# DO NOT CHANGE THE BELOW CODE WHICH WILL SERIALIZE THE ANSWERS FOR THE AUTOMATED TEST HARNESS TO LOAD AND MARK

	# write out all PERSON entries for character list for subtask 4
	writeHandle = codecs.open( 'characters.txt', 'w', 'utf-8', errors = 'replace' )
	if 'PERSON' in dictNE :
		for strNE in dictNE['PERSON'] :
			writeHandle.write( strNE.strip().lower()+ '\n' )
	writeHandle.close()

	# FILTER NE dict by types required for subtask 3
	listAllowedTypes = [ 'DATE', 'CARDINAL', 'ORDINAL', 'NORP' ]
	listKeys = list( dictNE.keys() )
	for strKey in listKeys :
		for nIndex in range(len(dictNE[strKey])) :
			dictNE[strKey][nIndex] = dictNE[strKey][nIndex].strip().lower()
		if not strKey in listAllowedTypes :
			del dictNE[strKey]

	# write filtered NE dict
	writeHandle = codecs.open( 'ne.json', 'w', 'utf-8', errors = 'replace' )
	strJSON = json.dumps( dictNE, indent=2 )
	writeHandle.write( strJSON + '\n' )
	writeHandle.close()

def exec_regex_toc( file_book = None ) :

	# INSERT CODE TO USE REGEX TO BUILD A TABLE OF CONTENTS FOR A BOOK (subtask 1)
	text = codecs.open(file_book,"r",encoding="utf-8-sig").read()

	chapters = re.findall(r'(?<=\s)^CHAPTER.*(?=\r)' ,text, flags=re.MULTILINE)
	r = re.compile(r'(\d+)(?:\.?)(.*)') # CHAPTER 1. XXXXXXXXXXXXXXX
	dictTOC = {re.search(r, chapter).group(1):re.search(r, chapter).group(2).strip() for chapter in chapters}

	# DO NOT CHANGE THE BELOW CODE WHICH WILL SERIALIZE THE ANSWERS FOR THE AUTOMATED TEST HARNESS TO LOAD AND MARK

	writeHandle = codecs.open( 'toc.json', 'w', 'utf-8', errors = 'replace' )
	strJSON = json.dumps( dictTOC, indent=2 )
	writeHandle.write( strJSON + '\n' )
	writeHandle.close()

def exec_regex_questions( file_chapter = None ) :

	# INSERT CODE TO USE REGEX TO LIST ALL QUESTIONS IN THE CHAPTER OF TEXT (subtask 2)
	text = codecs.open(file_chapter,"r",encoding="utf-8-sig").read()
	clean = re.sub('\r\n', ' ', text)

	#questions = re.findall(r'[A-Z][^.?!]*\?', clean)
	questions = re.findall(r'[A-Z][^.?!]*\?', clean)
	questions = [re.sub(r'.*\s‘', '', q) for q in questions]

	setQuestions = set(questions)

	# hardcoded output to show exactly what is expected to be serialized
	# setQuestions = set([
	# 	"Traddles?",
	# 	"And another shilling or so in biscuits, and another in fruit, eh?",
	# 	"Perhaps you’d like to spend a couple of shillings or so, in a bottle of currant wine by and by, up in the bedroom?",
	# 	"Has that fellow’--to the man with the wooden leg--‘been here again?"
	# 	])

	# DO NOT CHANGE THE BELOW CODE WHICH WILL SERIALIZE THE ANSWERS FOR THE AUTOMATED TEST HARNESS TO LOAD AND MARK

	writeHandle = codecs.open( 'questions.txt', 'w', 'utf-8', errors = 'replace' )
	for strQuestion in setQuestions :
		writeHandle.write( strQuestion + '\n' )
	writeHandle.close()

if __name__ == '__main__':
	if len(sys.argv) < 4 :
		raise Exception( 'missing command line args : ' + repr(sys.argv) )
	ontonotes_file = sys.argv[1]
	book_file = sys.argv[2]
	chapter_file = sys.argv[3]

	logger.info( 'ontonotes = ' + repr(ontonotes_file) )
	logger.info( 'book = ' + repr(book_file) )
	logger.info( 'chapter = ' + repr(chapter_file) )

	# DO NOT CHANGE THE CODE IN THIS FUNCTION

	#
	# subtask 1 >> extract chapter headings and create a table of contents from a provided plain text book (from www.gutenberg.org)
	# Input >> www.gutenberg.org sourced plain text file for a whole book
	# Output >> toc.json = { <chapter_number_text> : <chapter_title_text> }
	#

	exec_regex_toc( book_file )

	#
	# subtask 2 >> extract every question from a provided plain text chapter of text
	# Input >> www.gutenberg.org sourced plain text file for a chapter of a book
	# Output >> questions.txt = plain text set of extracted questions. one line per question.
	#

	exec_regex_questions( chapter_file )

	#
	# subtask 3 (NER) >> train NER using ontonotes dataset, then extract DATE, CARDINAL, ORDINAL, NORP entities from a provided chapter of text
	# Input >> www.gutenberg.org sourced plain text file for a chapter of a book
	# Output >> ne.json = { <ne_type> : [ <phrase>, <phrase>, ... ] }
	#
	# subtask 4 (text classifier) >> compile a list of characters from the target chapter
	# Input >> www.gutenberg.org sourced plain text file for a chapter of a book
	# Output >> characters.txt = plain text set of extracted character names. one line per character name.
	#

	exec_ner( chapter_file, ontonotes_file )

