import math
docs = [
    "solar solar forecasting",
    "solar energy",
    "wind energy"
]

#1.tokenize each doc
tokenized_docs=[]
for sent in docs:
    text = sent.lower().split()
    tokenized_docs.append(text)
#2.make the vocab
vocab =[]
for sent in tokenized_docs:
    for word in sent:
        if word not in vocab:
            vocab.append(word)
        
print("Vocabulary: ", vocab)

#3.calculate doc frequency
# The objective is to count the number of documents containing each word, 
# not the number of times each word occurs.

doc_freq={}
for word in vocab:
    doc_freq[word]=0

for sent in tokenized_docs:
    unique_words=[]
    for word in sent:
        if word not in unique_words:
            unique_words.append(word)

    for word in unique_words:
        doc_freq[word]+=1

N =len(docs)

#4.calculate term frequency
tfidf_matrix=[]
for sent in tokenized_docs:
    vector=[]
    doc_len = len(sent)
    for token in vocab:

        term_cnt =0
        for word in sent:
            if (token == word):
                term_cnt+=1

        tf  = term_cnt / doc_len
        df = doc_freq[token]
        idf = math.log((N+1)/(df+1)) + 1
        vector.append(tf*idf)
    
    tfidf_matrix.append(vector)

print("TF-IDF matrix:")
for vector in tfidf_matrix:
    print([round(value, 3) for value in vector])