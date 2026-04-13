import pandas as pd
from lxml import etree
import xmltodict
import json

DATA_XML_PATH = "../DATA/test-2025/archehr-qa.xml"
DATA_KEY_PATH = "../DATA/test-2025/archehr-qa_key.json"
RESULT_PATH = "../preproc/TEST-2025/"

def reader ():
    with open(DATA_XML_PATH, 'r') as f:
        data = xmltodict.parse(f.read())
    with open(DATA_KEY_PATH,"r") as f:
        key = json.load(f)
    return data, key

def aur_test (testua):
    return testua.replace("\n"," ")

def data_esaldi_bikoteak (data): #Datako esaldiak eta erreferentzia-zenbakia
    bikoteak={}
    for kasua in data["annotations"]["case"]:
        bikoteak[kasua["@id"]]={}
        for esaldia in kasua["note_excerpt_sentences"]["sentence"]:
            bikoteak[kasua["@id"]][esaldia["@id"]]=aur_test(esaldia["#text"])
    return bikoteak

def esaldi_erref_bikoteak (key): #Erantzunetako esladiak erreferentziekin elkartuta
    bikoteak={}
    for kasua in key:
        bikotea={}
        erantzuna=kasua["clinician_answer_without_citations"]
        esaldiak=erantzuna.split(". ")
        for en, esaldia in enumerate(esaldiak):
            zatiak=esaldia.split("[")
            if len(zatiak) == 1:
                if en == len(esaldiak)-1:
                    testua=zatiak[0]
                else:
                    testua=zatiak[0]+"."
                bikotea[en+1]={"Testua": testua, "Erref": []}
            else:
                testua=zatiak[0][:-1]+"."
                if en == len(esaldiak)-1:
                    erref=json.loads("["+zatiak[1][:-1])
                else:
                    erref=json.loads("["+zatiak[1])
                bikotea[en+1]={"Testua": testua, "Erref": erref}
        bikoteak[kasua["case_id"]]=bikotea
    return bikoteak

def data_oinarrizkoak (key): #Datako esaldiak eta horien oinarrizkotasuna
    bikoteak={}
    for kasua in key:
        tartekoa={}
        for esaldia in kasua["answers"]:
            tartekoa[esaldia["sentence_id"]]=esaldia["relevance"]
        bikoteak[kasua["case_id"]]=tartekoa
    return bikoteak

def paziente_galderak (data):
    bikoteak={}
    for kasua in data["annotations"]["case"]:
        bikoteak[kasua["@id"]]=kasua["clinician_question"]
    return bikoteak

def erantzun_txantiloi_sortzailea (erantzun_erref):
    txantiloia=[]
    for kasua in erantzun_erref:
        esaldiak=[]
        for esaldia in erantzun_erref[kasua]:
            esaldiak.append({"answer_id": esaldia, "evidence_id": erantzun_erref[kasua][esaldia]["Erref"]})
        txantiloia.append({"case_id": kasua, "prediction": esaldiak})
    return txantiloia

def saver (erantzun_erref, galdera_erref, galdera_oinarriz, gold, paziente_galdera_kli):
    with open(RESULT_PATH+"erantzunak.json", 'w') as f:
        json.dump(erantzun_erref,f)
    with open(RESULT_PATH+"ohar_klinikoa.json", 'w') as f:
        json.dump(galdera_erref,f)
    with open(RESULT_PATH+"ohar_klinikoa_oinarriz.json", 'w') as f:
        json.dump(galdera_oinarriz,f)
    with open(RESULT_PATH+"erantzunak_gold.json", 'w') as f:
        json.dump(gold,f)
    with open(RESULT_PATH+"galderak.json", 'w') as f:
        json.dump(paziente_galdera_kli,f)   

def main ():
    galdera_oinarriz = [ ]
    print("--AURREPROZESAKETA PROGRAMA--")
    data, key = reader ()
    print("Fitxategiak topatuta.")
    erantzun_erref=esaldi_erref_bikoteak(key)
    galdera_erref=data_esaldi_bikoteak(data)
    #galdera_oinarriz=data_oinarrizkoak(key)
    gold=erantzun_txantiloi_sortzailea (erantzun_erref)
    paziente_galdera_kli=paziente_galderak(data)
    print("Aurreprozesaketa eginda.")
    saver(erantzun_erref,galdera_erref,galdera_oinarriz,gold,paziente_galdera_kli)
    print("ONDO. Aurreprozesaketa gordeta.")


if __name__=="__main__":
    main()
