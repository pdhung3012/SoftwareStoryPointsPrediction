import stanza

def addDependenciesToSentence(docObj):
    lstSentences=docObj.sentences
    lstOutput=[]
    for sen in lstSentences:
        depends=sen._dependencies
        lstDepInfo=[]
        # depends=dict(depends)
        for deKey in depends:
            strElement=' '.join([deKey[2].text,deKey[0].text,deKey[1]])
            lstDepInfo.append(strElement)
        strDep=' '.join(lstDepInfo)
        lstOutput.append((strDep))
    strResult=' '.join(lstOutput)
    return strResult

# stanza.download('en')   # This downloads the English models for the neural pipeline
nlp = stanza.Pipeline() # This sets up a default neural pipeline in English
for index in range(0,100):
    doc = nlp("I hate you.  You don't work hard.")
    print('{}\t{}'.format(index,len(doc.sentences)))
    # doc.sentences[1].print_dependencies()
    # print(type(doc.sentences[0]))
    # print(doc.sentences[0]._dependencies[0][0].text)
    strOut=addDependenciesToSentence(doc)
    lstWords=doc.sentences[0]._words
    for item in lstWords:
        print(item.upos)
    print(strOut)