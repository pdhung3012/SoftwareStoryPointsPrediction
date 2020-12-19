from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence

# initialize the word embeddings
glove_embedding = WordEmbeddings('glove')
flair_embedding_forward = FlairEmbeddings('news-forward')
flair_embedding_backward = FlairEmbeddings('news-backward')

# # initialize the document embeddings, mode = mean
# document_embeddings = DocumentPoolEmbeddings([glove_embedding,
#                                               flair_embedding_backward,
#                                               flair_embedding_forward])
#
# # create an example sentence
# sentence = Sentence('The grass is green . And the sky is blue .')
#
# # embed the sentence with our document embedding
# document_embeddings.embed(sentence)
#
# # now check out the embedded sentence.
# print(sentence.get_embedding())
#
# document_embeddings = DocumentPoolEmbeddings([glove_embedding,
#                                              flair_embedding_backward,
#                                              flair_embedding_backward],
#                                              pooling='min')
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings

glove_embedding = WordEmbeddings('glove')

document_lstm_embeddings = DocumentRNNEmbeddings([glove_embedding], rnn_type='LSTM')


sentence = Sentence('jkjkjkjk')

document_lstm_embeddings.embed(sentence)

print(sentence.get_embedding())
print(sentence.get_embedding()[0])
print('aaa'+str(sentence.get_embedding()[0].data.item()))

print(len(sentence.get_embedding()))