import pandas as pd
from lxml import etree
import xmltodict
import json
from tqdm import tqdm
import gc
import time

from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from typing import List, Dict
import difflib
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import HumanMessagePromptTemplate

########
#Adjust the paths accordingly.
########

with open('../../DATA/archehr-qa.xml', 'r') as f:
    data = xmltodict.parse(f.read())
with open("../../DATA/archehr-qa_key.json","r") as f:
    key = json.load(f)

def aur_test (testua):
    return testua.replace("\n"," ")

##############################

  #PREPROCESSING Functions.

##############################

#Clinical answers and references macther.
def esaldi_erref_bikoteak (key): 
    bikoteak={}
    for kasua in key:
        bikotea={}
        erantzuna=kasua["clinician_answer"]
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

#Clinical notes preprocesser.
def data_esaldi_bikoteak (data): #Datako esaldiak eta erreferentzia-zenbakia
    bikoteak={}
    for kasua in data["annotations"]["case"]:
        bikoteak[kasua["@id"]]={}
        for esaldia in kasua["note_excerpt_sentences"]["sentence"]:
            bikoteak[kasua["@id"]][esaldia["@id"]]=aur_test(esaldia["#text"])
    return bikoteak

#Clinical note and sentence relevance matcher. 
def data_oinarrizkoak (key): #Datako esaldiak eta horien oinarrizkotasuna
    bikoteak={}
    for kasua in key:
        tartekoa={}
        for esaldia in kasua["answers"]:
            tartekoa[esaldia["sentence_id"]]=esaldia["relevance"]
        bikoteak[kasua["case_id"]]=tartekoa
    return bikoteak
  
#Formatter for the clinical notes.
def galdera_formatu_emailea (galdera_erref,kasua):
    formatatuta=""
    if kasua == "-1":
        itera=galdera_erref
    else:
        itera=galdera_erref[kasua]
    for esaldia in itera:
        formatatuta=formatatuta+esaldia+". -> "+itera[esaldia]+"\n"
    return formatatuta

#Formatter for the references. 
def erantzun_formatu_emailea (erantzun_erref,kasua):
    formatatuta=""
    for esaldia in erantzun_erref[kasua]:
        formatatuta=formatatuta+"* "+erantzun_erref[kasua][esaldia]["Testua"]+" [??]"+"\n"
    return formatatuta

#Structurer for clinical notes.
def erantzun_txantiloi_sortzailea (erantzun_erref):
    txantiloia=[]
    for kasua in erantzun_erref:
        esaldiak=[]
        for esaldia in erantzun_erref[kasua]:
            esaldiak.append({"answer_id": esaldia, "evidence_id": erantzun_erref[kasua][esaldia]["Erref"]})
        txantiloia.append({"case_id": kasua, "prediction": esaldiak})
    return txantiloia

#Clinical question loader.
def paziente_galderak (data):
    bikoteak={}
    for kasua in data["annotations"]["case"]:
        bikoteak[kasua["@id"]]=kasua["clinician_question"]
    return bikoteak

#Patient narrative loader.
def paziente_narratibak (data):
    bikoteak={}
    for kasua in data["annotations"]["case"]:
        bikoteak[kasua["@id"]]=kasua["patient_narrative"]
    return bikoteak

#Relevance codifier. 
def itxuraldatu_oinarriz (galdera_oinarriz,supplementary):
    esaldiak = { }
    for kasua in galdera_oinarriz:
        esaldiak[kasua]=[ ]
        for j in galdera_oinarriz[kasua]:
            if galdera_oinarriz[kasua][j] == "not-relevant":
                esaldiak[kasua].append(False)
            elif galdera_oinarriz[kasua][j] == "supplementary":
                esaldiak[kasua].append(supplementary)
            elif galdera_oinarriz[kasua][j] == "essential":
                esaldiak[kasua].append(True)
    return esaldiak

#Initialize prompts.
def prompt_loader (PROMPT_FILE):
    with open(PROMPT_FILE,"r") as f:
        prompt = f.read()
    return prompt

##############################

  #POSTPROCESSING Functions.

##############################

#Main postprocesser of model generations.
def postproz (emaitzak,galdera_erref):
    azkena=[ ]
    for kasua in emaitzak:
        azkena_dict={"case_id": kasua, "prediction": [ ]}
        testua = emaitzak[kasua]
        testua = testua.split("Your answer:")[1]
        lehen_esal=testua.split("\n")[1]
        zenbakiak = lehen_esal.split(",")
        for zat in zenbakiak:
            haut=re.sub('[^0-9]+','',zat)
            try:
                if int(haut) > len(galdera_erref) or int(haut) < 1:
                    continue
                azkena_dict["prediction"].append(int(haut))
            except:
                continue
        azkena_dict["prediction"]=set(azkena_dict["prediction"])
        azkena.append(azkena_dict)
    return azkena

#Evaluator of model generations. 
def ebaluatzailea (emaitzak,gold,galdera_erref,supplementary):
    iragar_all = [ ]
    urrezko_all = [ ]
    gold_list=itxuraldatu_oinarriz(gold,supplementary)
    for kasua in galdera_erref:
        num_sentences=len(galdera_erref[kasua])
        predicted=emaitzak[int(kasua)-1]["prediction"] #-9
        for j in range(num_sentences):
            if j+1 in predicted:
                iragar_all.append(True)
            else:
                iragar_all.append(False)
        urrezko_all.extend(gold_list[kasua]) 
    prec=precision_score(urrezko_all,iragar_all)    
    rec=recall_score(urrezko_all,iragar_all)
    f1=f1_score(urrezko_all,iragar_all)
    return prec, rec, f1

#Formatter for the results.
def format_converter (output):
    azkena = [ ]
    for en, i in enumerate(output):
        azkena.append({"case_id": en+1, "prediction": i})
    return azkena

#Formatter for the results. 
def def_converter (output):
    azkena = [ ]
    for en, i in enumerate(output):
        azkena.append({"case_id": str(en+1), "prediction": [str(x) for x in i]})
    return azkena

##############################

  #BATCH CREATORS AND WINDOWING FUNCTIONS.

##############################

#The function separates the sentence numbers from the explanations in the text generated by the model.
def txukundu_lehen_lerroa (lehen_lerroa,en,ATALASEA):
    zatiak=lehen_lerroa.split(",")
    muga_h=en*ATALASEA+1
    muga_b=en*(ATALASEA)+ATALASEA+1
    itzuli= [ ]
    for i in zatiak:
        unekoa=i.strip()
        if "." in unekoa:
            unekoa=unekoa.split(".")[0]
            if unekoa[0].isdigit() and int(unekoa[0]) >= muga_h and int(unekoa[0]) < muga_b and unekoa not in itzuli:
                itzuli.append(unekoa[0])
            if len(unekoa) > 1 and unekoa[1].isdigit() and int(unekoa[1]) >= muga_h and int(unekoa[1]) < muga_b and unekoa not in itzuli:
                itzuli.append(unekoa[1])
        elif "-" in unekoa:
            zat2=unekoa.split("-")
            if zat2[0].isdigit() and zat2[1].isdigit() and int(zat2[0]) >= muga_h and int(zat2[1]) < muga_b and unekoa not in itzuli:
                zer=list(range(int(zat2[0]),int(zat2[1])+1))
                s_zer=[str(i) for i in zer]
                itzuli.extend(s_zer)
        elif unekoa.isdigit() and int(unekoa) >= muga_h and int(unekoa) < muga_b and unekoa not in itzuli:
            itzuli.append(unekoa)
    return itzuli

#The function extracts the text generated by the model from the variable that contains the full prompt together with the generation. 
def extractor (result,keyword):
    zatiak = [ ]
    for atala in result:
        if keyword in atala:
            extracted=atala.split(keyword)[1].strip()
            zatiak.append(extracted)
        else:
            zatiak.append("")
            print("ERROR in the extraction")
    return zatiak

#This function converts the generation into a processable string, taking into account the windowing variable.
def list_converter (extracted,historia,ATALASEA):
    azkena = [ ]
    aurrekoa=historia[0]
    kont=0
    for en, iter in enumerate(extracted):
        final = [ ]
        unekoa=iter.split("\n")[0]
        komak=unekoa.split(",")
        if aurrekoa != historia[en]:
            aurrekoa=historia[en]
            kont=0
        for hau in komak:
            garbia=hau.strip()
            garbia=garbia.replace(".","")
            try:
                zenb=int(garbia)
                if zenb > kont*ATALASEA and zenb <= (kont+1)*ATALASEA:
                    final.append(zenb)
            except:
                continue
        kont+=1
        azkena.append(final)
    return azkena

#Creates the prompts for the coordinator model.
def coord_prompt_creator (essentials,nrevs,history,ATALASEA):
    prompts_ess=[ ]
    prompts_nrev=[ ]
    aurrekoa=history[0]
    prompta_ess=["-----Essential notes: ", "-----Reasoning: "]
    prompta_nrev=["-----Not-relevant notes: ", "-----Reasoning: "]
    zenb=0
    for en, i in enumerate(history):
        if aurrekoa != i:
            prompts_ess.append(prompta_ess[0]+"\n"+prompta_ess[1])
            prompts_nrev.append(prompta_nrev[0]+"\n"+prompta_nrev[1])
            prompta_ess=["-----Essential notes: ", "-----Reasoning: "]
            prompta_nrev=["-----Not-relevant notes: ", "-----Reasoning: "]
            zenb=0
        prompta_ess=elkartu_emaitzak(prompta_ess,essentials[en],zenb,ATALASEA)
        prompta_nrev=elkartu_emaitzak(prompta_nrev,nrevs[en],zenb,ATALASEA)
        aurrekoa=i
        zenb+=1
        #prompta_nrev=prompta_nrev+nrevs[en]+"\n"
    prompts_ess.append(prompta_ess[0]+"\n"+prompta_ess[1])
    prompts_nrev.append(prompta_nrev[0]+"\n"+prompta_nrev[1])
    return prompts_ess, prompts_nrev

#Creates the batches for the coordinator model.
def coord_batch_creator (templates,clin,sent,r1,r2,endwords):
    promptak=[ ]
    for atala in range(len(clin)):
        #print(sent)
        prompt = templates["coordinator"].invoke(
        {
            "clinical_question" : clin[str(atala+1)], #+9
            "sentences": galdera_formatu_emailea(sent[str(atala+1)],"-1"), #+9
            "r1": r1[atala],
            "r2": r2[atala],
            "assistant_response": endwords["coordinator"]
        })
        #print(prompt)
        promptak.append(prompt)
    return promptak

#Merges the generations of the model attenting to the windowing variable, to later input them to the coordinator. 
def elkartu_emaitzak (aurrekoak,emaitza,itera,ATALASEA):
    if "\n" in emaitza:
        zenbakiak=txukundu_lehen_lerroa(emaitza.split("\n")[0],itera,ATALASEA)
        zatiak='\n'.join(emaitza.split("\n")[1:])
    else:
        zenbakiak=""
        zatiak=emaitza
    aurrekoak[0]=aurrekoak[0]+','.join(zenbakiak)+","
    aurrekoak[1]=aurrekoak[1]+zatiak+"\n"
    return aurrekoak

#Merges the generations according to the windowing size. 
def elkartu_leihoak (essentials, history):
    elkartuta= [ ]
    aurrekoa=history[0]
    orain_artekoak = [ ]
    for i in range(len(history)):
        if history[i] != aurrekoa:
            elkartuta.append(orain_artekoak)
            orain_artekoak= [ ]
        bihur=essentials[i]
        orain_artekoak.extend(bihur)
        aurrekoa=history[i]
    elkartuta.append(orain_artekoak)
    return elkartuta

##############################

  #MODEL EXECUTOR AND HELPERS.

##############################

#Essential finder model
def essential_finder(prompt,endword):
    itzuli=extractor(llm.batch(prompt),endword)
    #print(itzuli)
    return itzuli

#Not relevant finder model
def nrev_finder(prompt,endword):
    itzuli=extractor(llm.batch(prompt),endword)
    #print(itzuli)
    return itzuli

#Windows creator
def threeshold_divisior (galdera_erref,narratives, clinicals, ATALASEA):
    pnar = [ ]
    pcli = [ ]
    divided= [ ]
    historia = [ ]
    for i in galdera_erref:
        kasua=galdera_erref[i]
        ohar_kop=len(kasua)
        j=-1
        for j in range(int(ohar_kop/ATALASEA)):
            unekoa=dict(list(kasua.items())[j*ATALASEA:(j+1)*ATALASEA])
            unekoa=galdera_formatu_emailea(unekoa,"-1")
            divided.append(unekoa)
            pnar.append(narratives[i])
            pcli.append(clinicals[i])
            historia.append(i)
        if int(ohar_kop/ATALASEA)*ATALASEA < ohar_kop:
            unekoa=dict(list(kasua.items())[(j+1)*ATALASEA:])
            unekoa=galdera_formatu_emailea(unekoa,"-1")
            divided.append(unekoa)
            pnar.append(narratives[i])
            pcli.append(clinicals[i])
            historia.append(i)
    return pnar, pcli, divided, historia

#Batch creator for the essential finder and the not-relevant finder.
def batch_creator (templates,parameters,endwords):
    nar=None
    cli=None
    sent=None
    essentials = [ ]
    nrev = [ ]
    if "patient_narrative" in parameters:
        nar=parameters["patient_narrative"]
    if "clinical_question" in parameters:
        cli=parameters["clinical_question"]
    if "sentences" in parameters:
        sent=parameters["sentences"]
    if nar is not None and cli is not None and sent is not None and "essential" in templates:
        for i in range(len(sent)):
            prompt=templates["essential"].invoke({"patient_narrative": nar[i],
                                           "clinical_question" : cli[i],
                                           "sentences": sent[i],
                                           "assistant_response": endwords["essential"]})
            essentials.append(prompt)
    if nar is not None and cli is not None and sent is not None and "not-relevant" in templates:
        for i in range(len(sent)):
            prompt=templates["not-relevant"].invoke({"patient_narrative": nar[i],
                                           "clinical_question" : cli[i],
                                           "sentences": sent[i],
                                           "assistant_response": endwords["not-relevant"]})
            nrev.append(prompt)
    return essentials, nrev
    
def zerrenda_txukundu (zerrenda,ATALASEA):
    azkena=""
    for en, i in enumerate(zerrenda):
        azkena=azkena+"From the note "+str(en*ATALASEA+1)+" to the sentence "+str((en+1)*ATALASEA)+":\n"+i
    return azkena
    
#Executor
def coordinator(templates,endwords,patient_nar,cli_quest,sent,ATALASEA):
    
    pnar, pcli, sent_div, historia = threeshold_divisior (sent,patient_nar,cli_quest,ATALASEA)
    #print(historia)

    batch_essent, batch_nrev = batch_creator(templates,{"patient_narrative": pnar, "clinical_question": pcli, "sentences": sent_div},endwords)

    #print(batch_essent[0])
    print("START")
    start=time.time()
    res_ess=essential_finder(batch_essent,endwords["essential"])
    end=time.time()
    print("ESSENTIAL: ",end-start)

    #print(res_ess)
    
    start=time.time()
    res_nrev=nrev_finder(batch_nrev,endwords["not-relevant"])
    end=time.time()
    print("REV:", end-start)

    
    
    r1, r2 = coord_prompt_creator(res_ess,res_nrev,historia,ATALASEA)
    
    batch_coord = coord_batch_creator (templates,cli_quest,sent,r1,r2,endwords)

    start=time.time()
    result = extractor(llm.batch(batch_coord),endwords["coordinator"])
    
    end=time.time()
    print("COORD: ",end-start)

    #print(result)
    
    result=list_converter(result,list(range(len(sent))),100)
    
    return result

def txukundu (erantzuna):
    emaitza = [ ]
    for en, i in enumerate(erantzuna):
        unekoa = {"case_id": str(en+1), "prediction": [ ]}
        for j in i:
            unekoa["prediction"].append(str(j))
        emaitza.append(unekoa)
    return emaitza

##############################

  #PERFORM THE PREPROCESSING.

##############################

erantzun_erref=esaldi_erref_bikoteak(key)
galdera_erref=data_esaldi_bikoteak(data)
galdera_oinarriz=data_oinarrizkoak(key)
gold=erantzun_txantiloi_sortzailea (erantzun_erref)
paziente_galdera_kli=paziente_galderak(data)
paziente_narratib=paziente_narratibak(data)

##############################

  #MODEL LOADER.

##############################

batch_size = 5

model_id = "HPAI-BSC/Llama3.1-Aloe-Beta-8B"
#model_id = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
tokenizer.pad_token_id = tokenizer.eos_token_id

# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto" )
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", temperature=0.70, do_sample=True)

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=500, 
    batch_size=batch_size,
    do_sample=True,
    model_kwargs={
        'device_map': 'auto',
        'batch_size':batch_size,
        "temperature": 0 
        }, 
)

llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={"temperature": 0.7}
                         ,batch_size=batch_size, verbose=True
                        )

##############################

  #PROMPT LOADERS.

##############################

system_template_esent = prompt_loader ("../prompts/essential_finder_prompt.txt")
system_template_nrev = prompt_loader ("../prompts/not_relevant_finder_prompt.txt")
system_template_coord = prompt_loader ("../prompts/coordinator_prompt.txt")

user_template="""
Patient Narrative:
{patient_narrative}

Clinical Question:
{clinical_question} 

Clinical Notes:
{sentences}
"""


user_template_coor="""
Clinical Question:
{clinical_question} 

Clinical Notes:
{sentences}

R1. I am a medical expert, member of your experts group. My task was to identify the essential sentences. 
{r1}

#R2. I am another medical expert, member of your experts group. My task was to identify the not-relevant sentences. 
#{r2}
"""

prompt_esent = ChatPromptTemplate.from_messages([
        ("system", system_template_esent),
        ("user",   user_template),
        ("assistant", "{assistant_response}")
    ]
)

prompt_nrev = ChatPromptTemplate.from_messages([
        ("system", system_template_nrev),
        ("user",   user_template),
        ("assistant", "{assistant_response}")
    ]
)

prompt_coord = ChatPromptTemplate.from_messages([
        ("system", system_template_coord),
        ("user",   user_template_coor),
        ("assistant", "{assistant_response}")
    ]
)

##############################

  #MAIN FUNCTION.

##############################

templates = {"essential": prompt_esent,
            "not-relevant": prompt_nrev,
            "coordinator": prompt_coord
            }
endwords = {"essential": "Essential notes::: ",
            "not-relevant": "Essential notes::: ",
            "coordinator": "List of essential notes::: "
           }
zerrenda = [ ]
ress_list = [ ]
resn_list = [ ]

ITERATIONS = 5 #Change the value of this variable if neccesary. Repeat all the process 5 times. 
for i in range(ITERATIONS):
    emaitzak=coordinator(templates,endwords,dict(list(paziente_narratib.items())),
                         dict(list(paziente_galdera_kli.items())),dict(list(galdera_erref.items())),15)
    em=ebaluatzailea(format_converter(emaitzak),dict(list(galdera_oinarriz.items())),dict(list(galdera_erref.items())),False)
    #print(emaitzak)
    zerrenda.append(em)
    ress_list.append(txukundu(emaitzak))
    bb=np.mean(pd.DataFrame(zerrenda),axis=0)
    print(bb[0],bb[1],bb[2])

with open ("results.json","w") as f:
    json.dump(ress_list,f)
