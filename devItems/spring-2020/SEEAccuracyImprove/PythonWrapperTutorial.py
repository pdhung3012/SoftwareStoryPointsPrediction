import logging
from stanfordcorenlp import StanfordCoreNLP
import json
import stanfordnlp
# stanfordnlp.download('en')
fpHost='http://localhost'
# Debug the wrapper
# nlp = StanfordCoreNLP(r'path_or_host', logging_level=logging.DEBUG)

# Check more info from the CoreNLP Server 
nlp = StanfordCoreNLP(fpHost,port=9000, logging_level=logging.DEBUG,  memory='8g')
sentence='I love Jenny. Her car is beautiful. '
print(nlp.word_tokenize(sentence)[0])
# print(nlp.pos_tag(sentence))
# # print(nlp.ner(sentence))
# print(nlp.parse(sentence))
# print(nlp.annotate(sentence,properties={
#                        'annotators': 'lemma'
#
#                    }))
sentences=nlp.annotate(sentence,properties={'annotators': 'lemma'})
jsonObj=json.loads(sentences)
print(jsonObj['sentences'][0]['tokens'][1]['lemma'])

print(nlp.dependency_parse(sentence))
nlp.close()