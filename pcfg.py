import nltk
from nltk import PCFG
from nltk.parse import ViterbiParser
pcfg=PCFG.fromstring("""
S ->NP VP[1.0]
PP ->P NP [1.0]  
NP  -> Det N[0.5]|Det N PP[0.25]|'I'[0.25]
VP ->V NP[0.5]|VP PP[0.5]
Det ->'an'[0.5]|'my' [0.5]
N ->'elephant'[0.5]|'pajamas'[0.5]
V  ->  'shot'[1.0]
P ->'in'[1.0]
   """ )
s="I shot an elephant in my pajamas"
tk=nltk.word_tokenize(s)
parser=ViterbiParser(pcfg)
for tree in parser.parse(tk):
    print(tree)
