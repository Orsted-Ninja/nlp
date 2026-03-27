import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.chunk import RegexpParser

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

text="The quick brown fox jumped over the lazy dog."
words=word_tokenize(text)
tagged_words=pos_tag(words)

grammar="""
NP: {<DT>?<JJ>*<NN.*>}   # Noun Phrase: optional determiner (DT), any number of adjectives (JJ), and a noun (NN, NNS, NNP)
VP: {<VB.*><NP|PP|CLAUSE>+$} # Verb Phrase: verb (VB, VBD, VBG...) followed by one or more noun phrases (NP), prepositional phrases (PP), or clauses (CLAUSE)
PP: {<IN><NP>}
             # Prepositional Phrase: preposition (IN) followed by a noun phrase (NP)
"""

chunk_parser=RegexpParser(grammar)
chunked=chunk_parser.parse(tagged_words)

print(chunked)
chunked.draw()
