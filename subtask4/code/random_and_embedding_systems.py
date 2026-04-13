import pandas as pd
from lxml import etree
import xmltodict
import json
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
import torch
from torch import cosine_similarity
import numpy as np

from typing import List, Dict
import difflib
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import HumanMessagePromptTemplate

##########################################
    #Random Systems: for baseline use.
##########################################

#All sentences without citations.
def ausazkoa_hutsak (gold):
    erantzuna=[]
    for kasua in gold:
        er={"case_id": kasua["case_id"], "prediction": [ ]}
        for galdera in kasua["prediction"]:
            er["prediction"].append({"answer_id": galdera["answer_id"], "evidence_id": [ ]})
        erantzuna.append(er)
    return erantzuna

#All sentences with all possible citations.
def ausazkoa_denak_beteta (gold,galdera_erref):
    erantzuna=[]
    for kasua in gold:
        er={"case_id": kasua["case_id"], "prediction": [ ]}
        for galdera in kasua["prediction"]:
            er["prediction"].append({"answer_id": galdera["answer_id"], "evidence_id": list(range(1,len(galdera_erref[kasua["case_id"]])+1))})
        erantzuna.append(er)
    return erantzuna

#All sentences with a random number of random citations.
def ausazkoa_zenbat_jakin_gabe (gold,galdera_erref):
    erantzuna=[]
    for kasua in gold:
        er={"case_id": kasua["case_id"], "prediction": [ ]}
        for galdera in kasua["prediction"]:
            kopurua=np.random.randint(0,len(galdera_erref[kasua["case_id"]])+1)
            ebidentziak=[np.random.randint(1,len(galdera_erref[kasua["case_id"]])+1) for i in range(kopurua)]
            er["prediction"].append({"answer_id": galdera["answer_id"], "evidence_id": ebidentziak})
        erantzuna.append(er)
    return erantzuna

#All sentences with random citations, but knowing the correct number of citations of each sentence.
def ausazkoa_zenbat_jakinda (gold,galdera_erref):
    erantzuna=[]
    for kasua in gold:
        er={"case_id": kasua["case_id"], "prediction": [ ]}
        for galdera in kasua["prediction"]:
            kopurua=len(galdera["evidence_id"])
            ebidentziak=[np.random.randint(1,len(galdera_erref[kasua["case_id"]])+1) for i in range(kopurua)]
            er["prediction"].append({"answer_id": galdera["answer_id"], "evidence_id": ebidentziak})
        erantzuna.append(er)
    return erantzuna

#Function to evaluate the results provided by the random systems or any other systems.
def ebaluatzailea (gold,erantzunak):
    emaitzak={}
    for e, kasua in enumerate(gold):
        tp=0
        fp=0
        fn=0
        error=0
        sent_error=0
        full=0
        esal=0
        hutsak=0
        hutsak_ondo=0
        handi_ondo=0
        handi=0
        if erantzunak[e]["prediction"] is None:
            for i in kasua["prediction"]:
                fn_cur+=len(i["evidence_id"])
            error=1
        else:
            for i in kasua["prediction"]:
                esal_id_gold=i["answer_id"]
                badago=0
                for j in erantzunak[e]["prediction"]:
                    if j["answer_id"] == esal_id_gold and j["evidence_id"] is not None:
                        tp_cur=len(set(i["evidence_id"]) & set(j["evidence_id"]))
                        fp_cur=len(set(j["evidence_id"]))-tp_cur
                        fn_cur=len(set(i["evidence_id"]))-tp_cur
                        if fp_cur == 0 and fn_cur == 0:
                            full+=1
                        if len(set(i["evidence_id"])) == 0: 
                            if len(set(j["evidence_id"])) == 0:
                                hutsak_ondo+=1
                            hutsak+=1
                        if len(set(i["evidence_id"])) >= 3:
                            if fp_cur == 0 and fn_cur == 0:
                                handi_ondo+=1
                            handi+=1
                        badago=1
                if badago == 0:
                    tp_cur=0
                    fp_cur=0
                    fn_cur=len(set(i["evidence_id"]))
                    if len(set(i["evidence_id"])) == 0:
                        hutsak+=1
                    if len(set(i["evidence_id"])) >= 3:
                        handi+=1
                    sent_error+=1
                esal+=1
                tp+=tp_cur
                fp+=fp_cur
                fn+=fn_cur
        emaitzak[kasua["case_id"]]={"TP": tp, "FP": fp, "FN": fn, "Error": error, "Sent_error": sent_error, "Full_ok": full, "Esal": esal, 
                                    "Void": hutsak, "Void_ok": hutsak_ondo, "Big": handi, "Big_ok": handi_ondo}
    return emaitzak

#Evaluator function: calculates some metrics used in the Shared Task. 
def micro_ebal (ebal):
    df=pd.DataFrame(ebal.values())
    if (sum(df["TP"])+sum(df["FN"])) != 0:
        micro_prec=sum(df["TP"])/(sum(df["TP"])+sum(df["FN"]))
    else:
        micro_prec=0
    if sum(df["TP"])+sum(df["FP"]) != 0:
        micro_rec=sum(df["TP"])/(sum(df["TP"])+sum(df["FP"]))
    else:
        micro_rec=0
    if (micro_prec+micro_rec) != 0:
        micro_f1=(2*(micro_prec*micro_rec))/(micro_prec+micro_rec)
    else:
        micro_f1=0
    return micro_prec, micro_rec, micro_f1

##########################################
    #Embedding Systems: for baseline use.
##########################################

#Cosine similarity calculator, based on the embeddings provided by the BERT models. 
def kalkulatu_bert_antzekotasuna (esal1,esal2,tokenizer,model):
    tok1 = tokenizer.tokenize(esal1)
    tok2 = tokenizer.tokenize(esal2)
    input_ids1 = torch.tensor(tokenizer.convert_tokens_to_ids(tok1)).unsqueeze(0)  # Batch size 1
    input_ids2 = torch.tensor(tokenizer.convert_tokens_to_ids(tok2)).unsqueeze(0)  # Batch size 1
    with torch.no_grad():
        outputs1 = model(input_ids1)
        outputs2 = model(input_ids2)
        embeddings1 = outputs1.last_hidden_state[:, 0, :]  # [CLS] token
        embeddings2 = outputs2.last_hidden_state[:, 0, :]  # [CLS] token
        
    similarity_score = cosine_similarity(embeddings1, embeddings2)
    return similarity_score

#BERT model executor.
def model_executor_bert (answers,clinical_notes,ATALASEA):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    erantzunak = [ ]
    for kasua in tqdm(answers):
        erantzuna = {"case_id": kasua, "prediction": [ ]}
        for esaldia in answers[kasua]:
            erantz_esal=answers[kasua][esaldia]["Testua"]
            ebidentziak = [ ]
            for g_esal in clinical_notes[kasua]:
                galder_esal=clinical_notes[kasua][g_esal]
                sim = kalkulatu_bert_antzekotasuna(erantz_esal,galder_esal,tokenizer,model)
                if sim > ATALASEA:
                    ebidentziak.append(int(g_esal))
            erantzuna["prediction"].append({"answer_id": esaldia, "evidence_id": ebidentziak})
        erantzunak.append(erantzuna)
        
    return erantzunak

#MedBERT model executor.
def model_executor_medbert (answers,clinical_notes,ATALASEA):
    tokenizer = BertTokenizer.from_pretrained('Charangan/MedBERT')
    model = BertModel.from_pretrained('Charangan/MedBERT')
    
    erantzunak = [ ]
    for kasua in tqdm(answers):
        erantzuna = {"case_id": kasua, "prediction": [ ]}
        for esaldia in answers[kasua]:
            erantz_esal=answers[kasua][esaldia]["Testua"]
            ebidentziak = [ ]
            for g_esal in clinical_notes[kasua]:
                galder_esal=clinical_notes[kasua][g_esal]
                sim = kalkulatu_bert_antzekotasuna(erantz_esal,galder_esal,tokenizer,model)
                if sim > ATALASEA:
                    ebidentziak.append(int(g_esal))
            erantzuna["prediction"].append({"answer_id": esaldia, "evidence_id": ebidentziak})
        erantzunak.append(erantzuna)
        
    return erantzunak

############################################
      #Preprocess the files.
############################################

with open('../DATA/archehr-qa.xml', 'r') as f:
    data = xmltodict.parse(f.read())
with open("../DATA/archehr-qa_key.json","r") as f:
    key = json.load(f)

erantzun_erref=esaldi_erref_bikoteak(key)
galdera_erref=data_esaldi_bikoteak(data)
galdera_oinarriz=data_oinarrizkoak(key)
gold=erantzun_txantiloi_sortzailea (erantzun_erref)
paziente_galdera_kli=paziente_galderak(data)

#############################################
  #RANDOM FUNCTIONS: Generate results
#############################################

emaitzak = [ ]
for i in range(1000):
    ebal_zenbat_jakin_gabe=ebaluatzailea(ausazkoa_zenbat_jakin_gabe(gold,galdera_erref),gold)
    emaitzak.append(micro_ebal(ebal_zenbat_jakin_gabe))
pp_bb=float(np.mean(pd.DataFrame(emaitzak)[0]))
rr_bb=float(np.mean(pd.DataFrame(emaitzak)[1]))
f1_bb=float(np.mean(pd.DataFrame(emaitzak)[2]))
pp_ma=float(np.max(pd.DataFrame(emaitzak)[0]))
rr_ma=float(np.max(pd.DataFrame(emaitzak)[1]))
f1_ma=float(np.max(pd.DataFrame(emaitzak)[2]))
print(pp_bb, rr_bb, f1_bb, pp_ma, rr_ma, f1_ma)

#############################################
  #EMBEDDING MODELS: Generate results
#############################################

#0.60 indicates the acceptance threshold. If the cosine similarity between two sentences is higher than that number, it is assumed that one sentence is based on the other. 
emaitza = model_executor_bert(erantzun_erref,galdera_erref,0.60)
emaitza2 = model_executor_medbert(erantzun_erref,galdera_erref,0.60)

ebal=ebaluatzailea(emaitza,gold)
micro_ebal(ebal)
print("BERT;",ebal)

ebal=ebaluatzailea(emaitza2,gold)
micro_ebal(ebal)
print("MedBERT:",ebal)




