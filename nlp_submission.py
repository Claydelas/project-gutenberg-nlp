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

def get_tagged_sentences(ontonotes: dict, max_sentences: int = 25000):
    tagged_sentences = []
    for sentences in ontonotes.values():
        for sentence in sentences.values():
            if len(tagged_sentences) >= max_sentences:
                return tagged_sentences
            pos = sentence.get('pos')
            if 'XX' in pos:
                continue
            if 'VERB' in pos:
                continue
            tokens = sentence.get('tokens')
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
                        tagged_sentence.append(
                            (tokens[token], pos[token], 'O'))
                    else:
                        tokens_arr = entity.get('tokens')
                        tag = entity.get('type')
                        #if len(tokens_arr) == 1:
                            #tagged_sentence.append(
                                #(tokens[token], pos[token], tag + "-S"))
                        if token == tokens_arr[0]:
                            tagged_sentence.append(
                                (tokens[token], pos[token], tag + "-B"))
                        #elif token == tokens_arr[-1]:
                            #tagged_sentence.append(
                                #(tokens[token], pos[token], tag + "-E"))
                        else:
                            tagged_sentence.append(
                                (tokens[token], pos[token], tag + "-I"))
                if tagged_sentence: tagged_sentences.append(tagged_sentence)
    return tagged_sentences


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


weekday_pat = re.compile(r"(mon|tues|wednes|thurs|fri|satur|sun)days?", re.IGNORECASE)
months_pat = re.compile(r'January|February|March|April|May|June|July|August|September|October|November|December', re.IGNORECASE)
temporal_pat = re.compile(r'now|past|once|present|days?|months?|years?', re.IGNORECASE)

TITLE_RE_PAT = re.compile(r'(Judge|Mr|Mrs|Ms|Miss|Drs?|Profs?|Sens?|Reps?|Attys?|Lt|Col|Gen|Messrs|Govs?|Adm|Rev|Maj|Sgt|Cpl|Pvt|Capt|Ave|Pres|Lieut|Hon|Brig|Co?mdr|Pfc|Spc|Supts?|Det|Mt|Ft|Adj|Adv|Asst|Assoc|Ens|Insp|Mlle|Mme|Msgr|Sfc)\.?', re.IGNORECASE)

ordinalPattern_java = re.compile("(?:(?:first|second|third|fourth|fifth|"+
                                       "sixth|seventh|eighth|ninth|tenth|"+
                                       "eleventh|twelfth|thirteenth|"+
                                       "fourteenth|fifteenth|sixteenth|"+
                                       "seventeenth|eighteenth|nineteenth|"+
                                       "twenty|twentieth|thirty|thirtieth|"+
                                       "forty|fortieth|fifty|fiftieth|"+
                                       "sixty|sixtieth|seventy|seventieth|"+
                                       "eighty|eightieth|ninety|ninetieth|"+
                                       "one|two|three|four|five|six|seven|"+
                                       "eight|nine|hundred|hundredth)-?)+|[0-9]+(?:st|nd|rd|th)", re.IGNORECASE)
 
CARDINAL_NUMBER_RE = r'^\s*(?:[+-]?)(?=\d|\.\d)\d*(?:\.\d*)?(?:[Ee](?:[+-]?\d+))?\s*'
 
ORDINAL_NUMBER_RE  = r'^\s*(?:[+-]?)(?=\d|\.\d)\d*(?:\.\d*)?(?:[Ee](?:[+-]?\d+))?(?:st|nd|rd|th)\s*$'

def isordinal(word):
    if re.match(ORDINAL_NUMBER_RE, word) or re.match(ordinalPattern_java, word):
        return True
 
    return False

def iscardinal(word):
    if re.match(CARDINAL_NUMBER_RE, word):
        return True

    return False;

wnpos = lambda e: ('a' if e[0].lower() == 'j' else e[0].lower()) if e[0].lower() in ['n', 'r', 'v'] else 'n'
wn = nltk.stem.WordNetLemmatizer()

stopwords = nltk.corpus.stopwords.words('english')

def word_shape(word):
    shape = re.sub(r'[A-Z]', 'X', word)
    shape = re.sub(r'[a-z]', 'x', shape)
    return re.sub(r'\d', 'd', shape)
                   

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isnumeric(),
        'postag': postag,
        'word.lemma': wn.lemmatize(word, wnpos(postag)),
        'postag[:2]': postag[:2],
        'word:title': True if re.match(TITLE_RE_PAT, word) else False,
        'word:isordinal': isordinal(word),
        'word:ismonth': True if re.match(months_pat, word) else False,
        'word:temporal': True if re.match(temporal_pat, word) else False,
        'word:shape': word_shape(word),
        'word:stopword': word.lower() in stopwords,
    }
            
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.lemma': wn.lemmatize(word1, wnpos(postag1)),
            '-1:word.istitle()': word1.istitle(),
            '-1:postag': postag1,
            '-1:word[-3:]': word1[-3:],
            '-1:word[-2:]': word1[-2:],
            '-1:postag[:2]': postag1[:2],
            '-1:word:title': True if re.match(TITLE_RE_PAT, word1) else False,
            '-1:word:isordinal': isordinal(word1),
            '-1:word:posbigram': postag1 + ' ' + sent[i][1],
            '-1:word:shape': word_shape(word1),
            '-1:word:stopword': word1.lower() in stopwords,
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.lemma': wn.lemmatize(word1, wnpos(postag1)),
            '+1:word.istitle()': word1.istitle(),
            '+1:postag': postag1,
            '+1:word[-3:]': word1[-3:],
            '+1:word[-2:]': word1[-2:],
            '+1:postag[:2]': postag1[:2],
            '+1:word:posbigram': sent[i][1] + ' ' + postag1,
            '+1:word:shape': word_shape(word1),
            '+1:word:stopword': word1.lower() in stopwords,
        })
    else:
        features['EOS'] = True

    if i > 1:
        word2 = sent[i - 2][0]
        postag2 = sent[i - 2][1]
        features.update({
            '-2:word.istitle()': word2.istitle(),
            '-2:postag': postag2,
            '-2:postag[:2]': postag2[:2],
            '-2:word:posbigram': postag2 + ' ' + sent[i - 1][1],
            '-2:word:shape': word_shape(word2)
        })

    if i < len(sent) - 2:
        word2 = sent[i + 2][0]
        postag2 = sent[i + 2][1]
        features.update({
            '+2:word.istitle()': word2.istitle(),
            '+2:postag': postag2,
            '+2:postag[:2]': postag2[:2],
            '+2:word:posbigram': sent[i + 1][1] + ' ' + postag2,
            '+2:word:shape': word_shape(word2)
        })

        
    # if i > 1:
    #     word2 = sent[i - 2][0]
    #     postag2 = sent[i - 2][1]
    #     features.update({
    #         '-2:word.lemma': wn.lemmatize(word2.lower(), wnpos(postag2)),
    #         '-2:word.istitle()': word2.istitle(),
    #         '-2:word.isupper()': word2.isupper(),
    #         '-2:postag': postag2,
    #         '-2:word[-3:]': word2[-3:],
    #         '-2:word[-2:]': word2[-2:],
    #         '-2:postag[:2]': postag2[:2],
    #         '-2:word:title': True if re.match(TITLE_RE_PAT, word2) else False,
    #         '-2:word:isordinal': isordinal(word2),
    #         '-2:word:posbigram': postag2 + ' ' + sent[i - 1][1],
    #         '-2:word:shape': word_shape(word2),
    #         '-2:word:stopword': word2.lower() in stopwords,
    #         '-2:word[:1]': word2[:1],
    #         '-2:word[:2]': word2[:2],
    #     })
    # if i < len(sent) - 2:
    #     word2 = sent[i + 2][0]
    #     postag2 = sent[i + 2][1]
    #     features.update({
    #         '+2:word.lemma': wn.lemmatize(word2.lower(), wnpos(postag2)),
    #         '+2:word.istitle()': word2.istitle(),
    #         '+2:word.isupper()': word2.isupper(),
    #         '+2:postag': postag2,
    #         '+2:word[-3:]': word2[-3:],
    #         '+2:word[-2:]': word2[-2:],
    #         '+2:postag[:2]': postag2[:2],
    #         '+2:word:posbigram': sent[i + 1][1] + ' ' + postag2,
    #         '+2:word:shape': word_shape(word2),
    #         '+2:word:stopword': word2.lower() in stopwords,
    #         '+2:word[:1]': word2[:1],
    #         '+2:word[:2]': word2[:2],
    #     })

    return features


def load_dataset(ontonotes_file, max_sentences=20000):
    # load parsed ontonotes dataset
    dataset = codecs.open(ontonotes_file, 'r', 'utf-8',
                          errors='replace').read()
    ontonotes = json.loads(dataset)
    return get_tagged_sentences(ontonotes, max_sentences)


def extract_entities(model, chapter_sents):
    chapter_word_tokens = [nltk.word_tokenize(sent) for sent in chapter_sents]

    pos_tags = [nltk.pos_tag(word) for word in chapter_word_tokens]
    features = [sent2features(s) for s in pos_tags]
    ner_tags = model.predict(features)

    ents = []
    for i, ne in enumerate(ner_tags):
        sentence = [word.lower() for word in chapter_word_tokens[i]]
        pos_tag = [y for x, y in pos_tags[i]]
        ents.append(list(zip(sentence, pos_tag, ne)))
    ents_clean = [
        x for x in [[tup for tup in s if tup[2] not in ['O']] for s in ents] if x
    ]
    return [item for sublist in ents_clean for item in sublist]


def chunk_entities(entities):
    chunked = []
    for entity_idx in range(len(entities)):
        flat_ent = ['', '', '']
        if entities[entity_idx][2].endswith('-B'):
            if entities[entity_idx][0].startswith('\''):
                continue
            flat_ent[0] = entities[entity_idx][0]
            flat_ent[1] = entities[entity_idx][1]
            flat_ent[2] = re.sub('-B$', '', entities[entity_idx][2])
            i = entity_idx + 1
            while i < len(entities) and entities[i][2].endswith('-I'):
                flat_ent[0] = flat_ent[0] + ' ' + entities[i][0]
                flat_ent[1] = flat_ent[1] + ' ' + entities[i][1]
                i += 1
            if (flat_ent[0], flat_ent[2]) in [(x[0], x[2]) for x in chunked]:
                continue
            if not '\'' in flat_ent[0] and not flat_ent[0].endswith('.'):
                chunked.append(tuple(flat_ent))
    return chunked

def exec_ner( file_chapter = None, ontonotes_file = None ) :

    # INSERT CODE TO TRAIN A CRF NER MODEL TO TAG THE CHAPTER OF TEXT (subtask 3)

    sentences = load_dataset(ontonotes_file = ontonotes_file, max_sentences = 30000)
    
    X_train = [sent2features(s) for s in sentences]
    y_train = [sent2labels(s) for s in sentences]

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.3,
        c2=0.05,
        max_iterations=120,
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)

    # USING NER MODEL AND REGEX GENERATE A SET OF BOOK CHARACTERS AND FILTERED SET OF NE TAGS (subtask 4)
    chapter = codecs.open(file_chapter, 'r', 'utf-8',
                      errors='replace').read()

    chapter = re.sub('\r\n', '\n', chapter)
    chapter = re.sub('(?<!\n)\n(?!\n)', ' ', chapter)

    chapter_sents = []
    for s in nltk.sent_tokenize(chapter):
        s = re.sub(r'[’‘]', '\'', s)
        if '\n\n' in s:
            for sp in s.split('\n\n'):
                chapter_sents.append(sp.strip('\n'))
        else: chapter_sents.append(s)

    chapter_sents = chapter_sents[2:]
    entities = extract_entities(crf, chapter_sents)
    chunked_entities = chunk_entities(entities)

    ner_people = [e[0] for e in chunked_entities if e[2] == 'PERSON']

    title_name_pat  = re.compile(r'(Judge|Mr|Mrs|Ms|Miss|Drs?|Profs?|Sens?|Reps?|Attys?|Lt|Col|Gen|Messrs|Govs?|Adm|Rev|Maj|Sgt|Cpl|Pvt|Capt|Ave|Pres|Lieut|Hon|Brig|Co?mdr|Pfc|Spc|Supts?|Det|Mt|Ft|Adj|Adv|Asst|Assoc|Ens|Insp|Mlle|Mme|Msgr|Sfc)\.? (?!You)(\b[A-Z][a-z]+[ -]?(?![a-z]|\d))+')

    for sentence in chapter_sents:
        match = re.search(title_name_pat, sentence)
        if match:
            ne = match.group(0).strip('- ').lower()
            if ne not in ner_people:
                ner_people.append(ne)

    dictNE = {
            "CARDINAL": [e[0] for e in chunked_entities if e[2] == 'CARDINAL'],
            "ORDINAL": [e[0] for e in chunked_entities if e[2] == 'ORDINAL'],
            "DATE": [e[0] for e in chunked_entities if e[2] == 'DATE'],
            "NORP": [e[0] for e in chunked_entities if e[2] == 'NORP'],
            "PERSON": ner_people
        }

    logger.info('Script execution finished.')

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
    text = codecs.open(file_book,"r",encoding="utf-8").read()

    arabicNumerals = '\d+'
    romanNumerals = '(?=[MDCLXVI])M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})'
    numbers1_9 = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    numbers10_19 = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
    numberWordsByTens = ['twenty', 'thirty', 'forty', 'fifty', 'sixty',
                          'seventy', 'eighty', 'ninety']
    numbers_hyph = [f'{x}-{y}' for x in numberWordsByTens for y in numbers1_9]
    numbers_space = [f'{x} {y}' for x in numberWordsByTens for y in numbers1_9]
    numberWords = numbers_hyph + numbers_space + numberWordsByTens + numbers10_19 + numbers1_9
    numberWordsPat = '(' + '|'.join(numberWords) + ')'
    ordinalNumberWordsByTens = ['twentieth', 'thirtieth', 'fortieth', 'fiftieth', 
                                'sixtieth', 'seventieth', 'eightieth', 'ninetieth']
    ordinalNumberWords = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 
                          'seventh', 'eighth', 'ninth', 'twelfth', 'last'] + \
                         [numberWord + 'th' for numberWord in numbers10_19] + ordinalNumberWordsByTens
    ordinalsPat = '(the )?(' + '|'.join(ordinalNumberWords) + ')'
    enumeratorsList = [arabicNumerals, romanNumerals, numberWordsPat, ordinalsPat] 
    enumerators = '(' + '|'.join(enumeratorsList) + ')'
    chap = r'(?: *\**)chapter +' + enumerators + r'(?:\.+ *|-+ *|—+ *| *)*'
    pat = re.compile(chap, re.IGNORECASE)
    
    text_split = text.split('\r\n')
    def patch_chapter(c, title):
        while c+1 <= len(text_split) and text_split[c+1]:
            title = title + ' ' + text_split[c+1].strip()
            c += 1
        return title

    headings = []
    for i, line in enumerate(text_split):
        if i-1 >= 0 and not text_split[i-1] and pat.match(line) is not None: # previous line is empty and current line matches
            m = pat.match(line)
            heading = text_split[i][m.span()[1]:]
            if heading and heading != '*': # Chapter X SOMETHING
                headings.append((m.group(1), patch_chapter(i, heading.strip())))
            elif i+1 <= len(text_split) and text_split[i+1]: # Chapter X \n SOMETHING
                headings.append((m.group(1), patch_chapter(i+1, text_split[i+1].strip())))
            elif i+2 <= len(text_split) and text_split[i+2]: # Chapter X \n\n SOMETHING
                headings.append((m.group(1), patch_chapter(i+2, text_split[i+2].strip())))
            else: headings.append((m.group(1), '')) # Chapter X NO_TITLE

    dictTOC = {heading[0]:heading[1] for heading in headings}

    # DO NOT CHANGE THE BELOW CODE WHICH WILL SERIALIZE THE ANSWERS FOR THE AUTOMATED TEST HARNESS TO LOAD AND MARK

    writeHandle = codecs.open( 'toc.json', 'w', 'utf-8', errors = 'replace' )
    strJSON = json.dumps( dictTOC, indent=2 )
    writeHandle.write( strJSON + '\n' )
    writeHandle.close()

def exec_regex_questions( file_chapter = None ) :

    # INSERT CODE TO USE REGEX TO LIST ALL QUESTIONS IN THE CHAPTER OF TEXT (subtask 2)
    text = codecs.open(file_chapter,"r",encoding="utf-8").read()
    #clean = re.sub('\r\n', ' ', text)

    #questions = re.findall(r'((?<=‘)[A-Za-z][^.?!]*\?(?=’)|[A-Z][^.?!]*\?)', clean)
    #questions = [re.sub(r'.*\s‘(?!.*’)', '', q).strip() for q in questions]

    #questions = re.findall(r'[A-Z][^.?!]*\?', clean)
    #questions = [re.sub(r'.*\s‘', '', q).strip() for q in questions]

    clean = re.sub('\r\n', '\n', text)
    clean = re.sub('(?<!\n)\n(?!\n)', ' ', clean)
    
    questions = re.findall(r'(?:(?<=‘)|(?<= )|(?<=“)|(?<=\n))([A-Z][^.?!\n]*\?|[A-Z][^.?!\n]*\b[A-Z][a-z]\.[^.?!\n]*\?)', clean)
    for q in questions:
        match = re.search(r'(?<!^)(?<=‘)([A-Za-z][^.?!]*\?|[A-Z][a-z][^.?!]*\b[A-Z][a-z]\.[^.?!]*\?)', q)
        if match:
            questions.append(match.group(0))

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

