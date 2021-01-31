import spacy
from spacy.lang.en import English
import networkx as nx
import matplotlib.pyplot as plt
from spektral.data import Graph
from spektral.datasets import QM9


def initDefaultTextEnvi():
    nlp_model = spacy.load('en_core_web_sm')
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    return nlp_model,nlp

def getSentences(text,nlp):
    result=None
    try:
        document = nlp(text)
        result= [sent.string.strip() for sent in document.sents]
    except Exception as e:
        print('sone error occured {}'.format(str(e)))
    return result

'''
def getSentences(text):
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    document = nlp(text)
    return [sent.string.strip() for sent in document.sents]
'''

def printToken(token):
    print(token.text, "->", token.dep_)

def appendChunk(original, chunk):
    return original + ' ' + chunk

def isRelationCandidate(token):
    deps = ["ROOT", "adj", "attr", "agent", "amod"]
    return any(subs in token.dep_ for subs in deps)

def isConstructionCandidate(token):
    deps = ["compound", "prep", "conj", "mod"]
    return any(subs in token.dep_ for subs in deps)

def processSubjectObjectPairs(tokens):
    subject = ''
    object = ''
    relation = ''
    subjectConstruction = ''
    objectConstruction = ''
    for token in tokens:
#        printToken(token)
        if "punct" in token.dep_:
            continue
        if isRelationCandidate(token):
            relation = appendChunk(relation, token.lemma_)
        if isConstructionCandidate(token):
            if subjectConstruction:
                subjectConstruction = appendChunk(subjectConstruction, token.text)
            if objectConstruction:
                objectConstruction = appendChunk(objectConstruction, token.text)
        if "subj" in token.dep_:
            subject = appendChunk(subject, token.text)
            subject = appendChunk(subjectConstruction, subject)
            subjectConstruction = ''
        if "obj" in token.dep_:
            object = appendChunk(object, token.text)
            object = appendChunk(objectConstruction, object)
            objectConstruction = ''

    #print (subject.strip(), ",", relation.strip(), ",", object.strip())
    return (subject.strip(), relation.strip(), object.strip())

def processSentence(sentence,nlp_model):
    tokens = nlp_model(sentence)
    return processSubjectObjectPairs(tokens)

def returnIDOrGenNewID(content,dictVocab):
    result=-1
    if content in dictVocab.keys():
        result=dictVocab[content]
    else:
        result=len(dictVocab.keys())
        dictVocab[content]=result
    return result

def extractGraphKeyFromTriple(triples,dictVocab):
    G = nx.Graph()
    for triple in triples:
#        print('aaaa {}'.format(triple[0]))
        key0 = returnIDOrGenNewID(triple[0], dictVocab)
        key1 = returnIDOrGenNewID(triple[1], dictVocab)
        key2 = returnIDOrGenNewID(triple[2], dictVocab)
        G.add_node(key0)
        G.add_node(key1)
        G.add_node(key2)
        G.add_edge(key0, key1)
        G.add_edge(key1, key2)
    return G

'''
def printGraph(triples):
    G = nx.Graph()
    for triple in triples:
        G.add_node(triple[0])
        G.add_node(triple[1])
        G.add_node(triple[2])
        G.add_edge(triple[0], triple[1])
        G.add_edge(triple[1], triple[2])

    pos = nx.spring_layout(G)
    plt.figure()
    nx.draw(G, pos, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color='seagreen', alpha=0.9,
            labels={node: node for node in G.nodes()})
    plt.axis('off')
    plt.show()
'''

def extractGraphSpekFromText(content,dictVocab,nlp_model,nlp):
    g=None
    sentences = getSentences(text, nlp)
    # nlp_model = spacy.load('en_core_web_sm')

    triples = []
    print(text)
    for sentence in sentences:
        triples.append(processSentence(sentence, nlp_model))
    g = extractGraphKeyFromTriple(triples, dictVocab)
    adjacency_matrix = nx.adjacency_matrix(g)
    graphSpek = Graph()
    graphSpek.a = adjacency_matrix
    return graphSpek

if __name__ == "__main__":

    '''
    text = "London is the capital and largest city of England and the United Kingdom. Standing on the River " \
           "Thames in the south-east of England, at the head of its 50-mile (80 km) estuary leading to " \
           "the North Sea, London has been a major settlement for two millennia. " \
           "Londinium was founded by the Romans. The City of London, " \
           "London's ancient core − an area of just 1.12 square miles (2.9 km2) and colloquially known as " \
           "the Square Mile − retains boundaries that follow closely its medieval limits." \
           "The City of Westminster is also an Inner London borough holding city status. " \
           "Greater London is governed by the Mayor of London and the London Assembly." \
           "London is located in the southeast of England." \
           "Westminster is located in London." \
           "London is the biggest city in Britain. London has a population of 7,172,036."
    '''

    text = "London hated you and him. et move to a new city."

    nlp_model,nlp=initDefaultTextEnvi()
    dictVocab={}

    g=extractGraphSpekFromText(text,dictVocab,nlp_model,nlp)
    print('gaa {}\n\n{}'.format((g.a), g.y))
    print('len vocab {}'.format(len(dictVocab.keys())))

