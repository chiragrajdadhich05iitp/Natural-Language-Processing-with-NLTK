# Creating a single Python file that consolidates all the code snippets from the uploaded PDF.

code_content = """
# Natural Language Processing with NLTK - Consolidated Code

import os
import nltk
import warnings
warnings.filterwarnings('ignore')

# Tokenization Examples
from nltk.tokenize import word_tokenize, sent_tokenize, blankline_tokenize, WhitespaceTokenizer, wordpunct_tokenize
from nltk.util import bigrams, trigrams, ngrams
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk

# Sample Text
AI = '''AI, machine learning and deep learning are common terms in enterprise IT and sometimes used interchangeably, especially by companies in their marketing materials. But there are distinctions. The term AI, coined in the 1950s, refers to the simulation of human intelligence by machines. It covers an ever-changing set of capabilities as new technologies are developed. Technologies that come under the umbrella of AI include machine learning and deep learning. Machine learning enables software applications to become more accurate at predicting outcomes without being explicitly programmed to do so. Machine learning algorithms use historical data as input to predict new output values. This approach became vastly more effective with the rise of large data sets to train on. Deep learning, a subset of machine learning, is based on our understanding of how the brain is structured. Deep learning's use of artificial neural networks structure is the underpinning of recent advances in AI, including self-driving cars and ChatGPT.'''

# Word Tokenize
AI_tokens = word_tokenize(AI)
print("Word Tokens:", AI_tokens)
print("Number of Tokens:", len(AI_tokens))

# Sentence Tokenize
AI_sentences = sent_tokenize(AI)
print("Sentence Tokens:", AI_sentences)

# Blankline Tokenize (paragraphs)
AI_paragraphs = blankline_tokenize(AI)
print("Paragraph Tokens:", AI_paragraphs)
print("Number of Paragraphs:", len(AI_paragraphs))

# Whitespace Tokenize
wt = WhitespaceTokenizer().tokenize(AI)
print("Whitespace Tokens:", wt)
print("Number of Whitespace Tokens:", len(wt))

# WordPunct Tokenize
s = 'good apple cost $3.88 in hyderabad.please buy two of them. Thanks'
wp = wordpunct_tokenize(s)
print("WordPunct Tokens:", wp)

# N-grams
string = 'hello the best and most beautiful thing cannot be seen or touched, it should be felt'
str_tokens = word_tokenize(string)
print("Bigrams:", list(bigrams(str_tokens)))
print("Trigrams:", list(trigrams(str_tokens)))
print("5-grams:", list(ngrams(str_tokens, 5)))

# Stemming
pst = PorterStemmer()
lst = LancasterStemmer()
sst = SnowballStemmer('english')
word_lem = WordNetLemmatizer()

words_to_stem = ['give','given','giving','gave','thinking','maximum']
print("PorterStemmer:")
for word in words_to_stem:
    print(word + ':' + pst.stem(word))

print("LancasterStemmer:")
for word in words_to_stem:
    print(word + ':' + lst.stem(word))

print("SnowballStemmer:")
for word in words_to_stem:
    print(word + ':' + sst.stem(word))

# Lemmatization
nltk.download('wordnet')
print("WordNet Lemmatizer:")
for word in words_to_stem:
    print(word + ':' + word_lem.lemmatize(word))

# Stopwords
print("English Stopwords:", stopwords.words('english'))
print("French Stopwords:", stopwords.words('french'))
print("Number of English Stopwords:", len(stopwords.words('english')))
print("Number of French Stopwords:", len(stopwords.words('french')))

# POS Tagging
sent = 'sam is natural when it comes to drawings'
sent_tokens = word_tokenize(sent)
for token in sent_tokens:
    print(pos_tag([token]))

sent2 = 'john is eating a delicious cake'
sent2_tokens = word_tokenize(sent2)
for token in sent2_tokens:
    print(pos_tag([token]))

# Named Entity Recognition
nltk.download('maxent_ne_chunker')
nltk.download('words')

NE_sent = 'The US president stays in the WHITEHOUSE'
NE_tokens = word_tokenize(NE_sent)
NE_tags = pos_tag(NE_tokens)
NE_NER = ne_chunk(NE_tags)
print("Named Entity Chunking:", NE_NER)

# WordCloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = "Innovation drives progress as technology evolves rapidly across industries. Creativity, strategy, and collaboration empower teams to solve challenges with precision. Data, insight, growth, and impact are at the core of success. Agile methods, digital tools, smart systems, and scalable platforms redefine productivity. Visionary leaders inspire change through resilience, adaptability, and passion. From code to design, research to implementation, every idea shapes the future. Empower, transform, analyze, optimize, connect, deliver"
wordcloud = WordCloud(background_color='black').generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
"""

# Save to a .py file
file_path = "/mnt/data/nlp_nltk_full_code.py"
with open(file_path, "w") as f:
    f.write(code_content)

file_path
