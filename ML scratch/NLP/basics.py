import re


def tokens(sentences):
    tokenized_sent = []
    for sentence in sentences:
        tokens= sentence.lower().split()
        tokenized_sent.append(tokens)
    # print(tokenized_sent)
    return tokenized_sent

sentences = [
    "The models are learning",
    "A model is learning"
]

token_sent=  tokens(sentences=sentences)


#build the vocab
def vocab():
    vocabulary = []
    for sentence in token_sent:
        for token in sentence:
            if token not in vocabulary:
                vocabulary.append(token)

    # print(vocabulary)
    return vocabulary


vocabulary = vocab()


#This is a binary Bag-of-Words representation
#convert into vectors
vectors = []
for sent in token_sent:
    vector=[]
    for word in vocabulary:
        if word in sent:
            vector.append(1)
        else:
            vector.append(0)
    vectors.append(vector)

print(vectors)