import os, codecs, re
# for root, dirs, files in os.walk('chapters'):
#     for file in files:
#         text = codecs.open(os.path.join('chapters', file),"r",encoding="utf-8").read()
#         # clean = re.sub('\n', ' ', text)
#         # questions = re.findall(r'[A-Z][^.?!]*\?', clean)
#         # questions = [re.sub(r'.*\s‘', '', q).strip() for q in questions]
#         # questions2 = re.findall(r'((?<=‘)[A-Za-z][^.?!]*\?(?=’)|[A-Z][^.?!]*\?)', clean)
#         # questions2 = [re.sub(r'.*\s‘(?!.*’)', '', q).strip() for q in questions2]
#         # setQuestions = set(questions)
#         # setQuestions2 = set(questions2)

#         #setQuestions = set(questions)
#         #print(file, len(questions))
#         seen = []
#         for q in text.split('\n'):
#            m = re.search(r'.*\?', q)
#            if m:
#                q2 = m.group(0)[-12:]
#                if q2 not in seen: seen.append(q2)
#         #questions = len(re.findall(r'\?', clean))
#         questions = len(seen)
        #if questions >= 19:
            #print(questions)
        #print(file, len(setQuestions), len(setQuestions2), sep='\t')
# import nltk
# for root, dirs, files in os.walk('chapters'):
#     for file in files:
#         if file == 'CHAPTER XVI. TOO FULL OF ADVENTURE.txt':#'CHAPTER XIX. A PLEASANT DAY.txt':
#             text = codecs.open(os.path.join('chapters', file),"r",encoding="utf-8").read()

#             clean = re.sub('(?<!\n)\n(?!\n)', ' ', text)
#             questions3 = re.findall(r'(?:(?<=‘)|(?<= )|(?<=“)|(?<=\n))([A-Z][^.?!\n]*\?|[A-Z][^.?!\n]*\b[A-Z][a-z]\.[^.?!\n]*\?)', clean)

#             clean = re.sub('\n', ' ', text)
#             #print(clean)
#             questions = re.findall(r'[A-Z][^.?!]*\?', clean)
#             #questions_pat = r'(?<=‘| ‘| )[A-Z]([^.?!\n]*|[^.?!\n]*\b[A-Z][a-z]\.[^.?!\n]*)\?'
#             questionspost = [re.sub(r'.*\s‘', '', q).strip() for q in questions]

#             questions2 = re.findall(r'((?<=‘)[A-Za-z][^.?!]*\?(?=’)|[A-Z][^.?!]*\?)', clean)
#             questions2post = [re.sub(r'.*\s‘(?!.*’)', '', q).strip() for q in questions2]

#             left_intersection = set(questionspost).difference(set(questions2post))
#             right_intersection = set(questions2post).difference(set(questionspost))
#             print(file, len(left_intersection), len(right_intersection), sep=':')
#             print(left_intersection)
#             print(right_intersection)
#             print('-------------')
#             new_left_sec = set(questionspost).difference(set(questions3))
#             new_right_sec = set(questions3).difference(set(questionspost))
#             print(file, len(new_left_sec), len(new_right_sec), sep=':')
#             print(new_left_sec)
#             print(new_right_sec)

            #print(file, len(questionspost), len(questions2post), sep='\t')
class PunktLanguageVars(object):
    """
    Stores variables, mostly regular expressions, which may be
    language-dependent for correct application of the algorithm.
    An extension of this class may modify its properties to suit
    a language other than English; an instance can then be passed
    as an argument to PunktSentenceTokenizer and PunktTrainer
    constructors.
    """

    __slots__ = ("_re_period_context", "_re_word_tokenizer")

    def __getstate__(self):
        # All modifications to the class are performed by inheritance.
        # Non-default parameters to be pickled must be defined in the inherited
        # class.
        return 1

    def __setstate__(self, state):
        return 1

    sent_end_chars = (".", "?", "!")
    """Characters which are candidates for sentence boundaries"""

    @property
    def _re_sent_end_chars(self):
        return "[%s]" % re.escape("".join(self.sent_end_chars))

    internal_punctuation = "!?,:;"  # might want to extend this..
    """sentence internal punctuation, which indicates an abbreviation if
    preceded by a period-final token."""

    re_boundary_realignment = re.compile(r'["\')\]}’‘]+?(?:\s+|(?=--)|$)', re.MULTILINE)
    """Used to realign punctuation that should be included in a sentence
    although it follows the period (or ?, !)."""

    _re_word_start = r"[^\(\"\`{\[:;&\#\*@\)}\]\-,]"
    """Excludes some characters from starting word tokens"""

    @property
    def _re_non_word_chars(self):
        return r"(?:[)\";}\]\*:@\'\({\[%s])" % re.escape("".join(set(self.sent_end_chars) - {"."}))
    """Characters that cannot appear within words"""

    _re_multi_char_punct = r"(?:\-{2,}|\.{2,}|(?:\.\s){2,}\.|(?:‘[!?]\s*.*’))"
    """Hyphen and ellipsis are multi-character punctuation"""

    _word_tokenize_fmt = r"""(
        %(MultiChar)s
        |
        (?=%(WordStart)s)\S+?  # Accept word characters until end is found
        (?= # Sequences marking a word's end
            \s|                                 # White-space
            $|                                  # End-of-string
            %(NonWord)s|%(MultiChar)s|          # Punctuation
            ,(?=$|\s|%(NonWord)s|%(MultiChar)s) # Comma if at end of word
        )
        |
        \S
    )"""
    """Format of a regular expression to split punctuation from words,
    excluding period."""

    def _word_tokenizer_re(self):
        """Compiles and returns a regular expression for word tokenization"""
        try:
            return self._re_word_tokenizer
        except AttributeError:
            self._re_word_tokenizer = re.compile(
                self._word_tokenize_fmt
                % {
                    "NonWord": self._re_non_word_chars,
                    "MultiChar": self._re_multi_char_punct,
                    "WordStart": self._re_word_start,
                },
                re.UNICODE | re.VERBOSE,
            )
            return self._re_word_tokenizer

    def word_tokenize(self, s):
        """Tokenize a string to split off punctuation other than periods"""
        return self._word_tokenizer_re().findall(s)

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            \s+(?P<next_tok>\S+)     # or whitespace and some other token
        ))"""
    """Format of a regular expression to find contexts including possible
    sentence boundaries. Matches token which the possible sentence boundary
    ends, and matches the following token within a lookahead expression."""

    def period_context_re(self):
        """Compiles and returns a regular expression to find contexts
        including possible sentence boundaries."""
        try:
            return self._re_period_context
        except:
            self._re_period_context = re.compile(
                self._period_context_fmt
                % {
                    "NonWord": self._re_non_word_chars,
                    "SentEndChars": self._re_sent_end_chars,
                },
                re.UNICODE | re.VERBOSE,
            )
            return self._re_period_context

import nltk
extra_abbreviations = ['dr', 'vs', 'mr', 'mrs', 'prof', 'inc', 'i.e']
tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer(lang_vars=PunktLanguageVars())
tokenizer._params.abbrev_types.update(extra_abbreviations)

text = codecs.open("chapters/CHAPTER XVI. TOO FULL OF ADVENTURE.txt","r","utf8").read()
tokenizer.train(text)

#tokens = tokenizer.sentences_(
#    'Mr. Creakle whispered, ‘Hah! What’s this?’ and bent his eyes upon me, as if he would have burnt me up with them.', realign_boundaries=True)
#print(tokens)
#print(len(tokens))

#for grp in re.findall(r'‘[^‘]*(?<!Mr)\.’|(‘[^‘]*’)*([^’.]*(?<!Mr)\.)', 'Mr. Creakle whispered, ‘Hah! What’s this?’ and bent his eyes upon me, as if he would have burnt me up with them.'):
 #   print("".join(grp))

    #(?:‘[!?]\s*.*’)

#print(nltk.tokenize.word_tokenize('Mr. Creakle whispered, ‘Hah! What’s this?’ and bent his eyes upon me, as if he would have burnt me up with them.'))

clean = re.sub('(?<!\n)\n(?!\n)', ' ', text)

# pat = re.compile(r'‘([A-Z][a-z]*[!?.].*)*’(?!\n|s)')

# subs = []
# i=0
# while pat.search(clean):
#     match = pat.search(clean).group(0)
#     sub = f'<SUSPENDED_Q{i}>'
#     clean = re.sub(re.escape(match), sub, clean)
#     subs.append((match, sub))
#     i += 1

# tokenizer = nltk.tokenize.PunktSentenceTokenizer()
# extra_abbreviations = ['dr', 'vs', 'mr', 'mrs', 'prof', 'inc', 'i.e']
# tokenizer._params.abbrev_types.update(extra_abbreviations)
# sents = tokenizer.tokenize(clean)[1:]

# sents2 = []
# for s in sents:
#     for ss in subs:
#         sents2.append(re.sub(ss[1], ss[0], s))

# sents2

sents = []
for s in nltk.tokenize.sent_tokenize(clean):
    if '\n\n' in s:
        for sp in s.split('\n\n'):
            sents.append(sp)
    else: sents.append(s)
sents
#from nltk.corpus import stopwords
stopwords = nltk.corpus.stopwords.words('english')
stopwords

chapter = codecs.open("chapters/CHAPTER XVI. TOO FULL OF ADVENTURE.txt", 'r', 'utf-8',
                      errors='replace').read()
chapter = re.sub('\r\n', '\n', chapter)
chapter = re.sub('(?<!\n)\n(?!\n)', ' ', chapter)
chapter_sents = []
for s in nltk.sent_tokenize(chapter):
    s = re.sub(r'[’‘]', '\'', s)
    if '\n\n' in s:
        for sp in s.split('\n\n'):
            chapter_sents.append(sp)
    else: chapter_sents.append(s)
chapter_word_tokens = [nltk.word_tokenize(sent) for sent in chapter_sents]
print()