from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from tqdm import tqdm
import json
import time
import gc
import os
import pandas as pd
import argparse
import difflib
import numpy as np

parser = argparse.ArgumentParser(description='Multiple agent with different backbone LLMs caller function.')

parser.add_argument('-pf1', '--prompt1', help='Path to prompt file')
parser.add_argument('-pf2', '--prompt2', help='Path to prompt file')
parser.add_argument('-of', '--output', help='Path to output file')
parser.add_argument('-tf', '--time', help='Path to times file')
parser.add_argument('-mo1', '--model1', help='First model name')
parser.add_argument('-mo2', '--model2', help='Second model name')
parser.add_argument('-fp', '--float', help='Floating point')
parser.add_argument('-it1', '--iters1', help='Number of voting agents of the first kind')
parser.add_argument('-it2', '--iters2', help='Number of voting agents of the second kind')
parser.add_argument('-maj', '--majority', help='Percentage for majority')

args = parser.parse_args()

PREPROC_FILES = {"clinical_sentences": "../preproc/TEST-2025/ohar_klinikoa.json",
                 "patient_question": "../preproc/TEST-2025/galderak.json",
                 "answer_sentences": "../preproc/TEST-2025/erantzunak.json"}

GOLD_PATH = "../preproc/erantzunak_gold.json"

PROMPT_FILE1 = args.prompt1
PROMPT_FILE2 = args.prompt2

FLOAT_POINT = args.float

PROMPT_VARIABLES = ["clinical_sentences",
                    "patient_question", 
                    "answer_sentences"]

GPU_ID = 0

ITERS1 = int(args.iters1)
ITERS2 = int(args.iters2)

MAJORITY = float(args.majority)

MODEL_NAME1 = args.model1
MODEL_NAME2 = args.model2
#MODEL_NAME1 = "HPAI-BSC/Llama3.1-Aloe-Beta-8B"
#MODEL_NAME2 = "google/medgemma-4b-it"
#MODEL_NAME = "ibm-granite/granite-4.0-h-tiny"
#MODEL_NAME = "meta-llama/Llama-3.1-8B"
#MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"
#MODEL_NAME2 = "Qwen/Qwen3-8B"
#MODEL_NAME = "google/medgemma-4b-it"

MODEL_PARAMS = {"tokens": 256, 
                "temperature": 0.7, 
                "do_sample": True}

OUTPUT_FILE = args.output
TIMES_FILE = args.time

MULTIPLE = 1

#Function to give a structured format to the clinical answer.
def erantzun_formatu_emailea (erantzun_erref,kasua):
    formatatuta=""
    for esaldia in erantzun_erref[kasua]:
        formatatuta=formatatuta+"* "+erantzun_erref[kasua][esaldia]["Testua"]+" [??]"+"\n"
    return formatatuta

##########################################################################################

        #POSTPROCESSING FUNCTIONS: Transform the answer provided by the model into 
        #a processable answer in order to proceed with the evaluation.

##########################################################################################

#Function to calculate the length of the largest identical sequence of characters between two sentences. 
def seq_sim (sent1,sent2):
    sent1_v=sent1.split("[")[0]
    sent2_v=sent2.split("[")[0]
    seq=difflib.SequenceMatcher(a=sent1_v, b=sent2_v)
    rat = seq.ratio()
    return rat

#Function to calculate the similarity between two sentences using the previous function.
def antzekotasun_bilatzailea (esaldia,berezkoa,log,en):
    balioak=[]
    for ber_es in berezkoa:
        cs=seq_sim(esaldia,ber_es)
        balioak.append(cs)
    max_ind=int(np.array(balioak).argmax())+1
    max_val=max(balioak)
    if max_val > 0.95:
        return max_ind, log
    elif max_val > 0.5:
        log.append({"Sentence": en,
                    "Error": "Not copy: some distinct words"})
        return max_ind, log
    log.append({"Sentece": en,
                "Error": "Not copy: very distinct sentence"})
    return None, log

# Fuction to split the sentences of the clinical answer provided to the model, twying to correct some possible disconcordances in the format. 
def analyze_brackets (ir_esal,log,en,kar):
    salbatu=ir_esal.split("[")[1]
    salbatu="["+salbatu
    salbatu=salbatu.split("]")[0]
    jatorrizkoa=salbatu+"]"
    salbatu=salbatu+"]"
    if (ir_esal[0] != "*" and kar == "\n") or (ir_esal[1] != "*" and kar == "]"):
        log.append({"Sentence": en, 
                    "Error": "Missing *: "+ir_esal[:10]})
    if "-" in salbatu:
        salbatu=salbatu.replace("-",",")
        log.append({"Sentence": en, 
                    "Error": "Symbol -: "+jatorrizkoa})
    if "?" in salbatu:
        salbatu=salbatu.replace("?","")
        log.append({"Sentence": en, 
                    "Error": "Symbol ?: "+jatorrizkoa})
    return salbatu, log

#Function to remove duplicate sentences from the answer provided by the model. 
def eguneratu_emaitza (emaitza,max_ind,erref,log,en):
    eguneratu=False
    for en2, i in enumerate(emaitza):
        if i["answer_id"] == max_ind:
            eguneratu=True
            break
    if eguneratu and erref is not None:
        for j in erref:
            if emaitza[en2]["evidence_id"] is not None and j not in emaitza[en2]["evidence_id"]:
                emaitza[en2]["evidence_id"].append(j)
        log.append({"Sentence": en,
                    "Error": "Duplicate of the answer sentence "+str(max_ind)})
    else:
        emaitza.append({"answer_id": max_ind, "evidence_id": erref})
    return emaitza, log, eguneratu
    
#Function to postprocess the answer generated by the model.      
def esperotako_formatua (iragarpena,berezkoa,kar): 
    erantzunak=[ ]
    log=[ ]
    yes=0
    en=0
    irag_zatiak=iragarpena.split(kar)
    if len(irag_zatiak) > 1 or len(berezkoa) == 1:
        for ir_esal in irag_zatiak:
            en+=1
            if len(ir_esal) <= 1 or yes == 2:
                continue
            if "[" in ir_esal and (kar == "]" or "]" in ir_esal): #Reorder sentences, if neccesary
                max_ind, log=antzekotasun_bilatzailea(ir_esal,berezkoa,log,en)
                if max_ind is None:
                    continue
                yes=1
                salbatu, log = analyze_brackets(ir_esal,log,en,kar)
            else:
                if yes == 0:
                    continue
                else:
                    log.append({"Sentence": en, 
                                "Error": "No brackets: "+ir_esal[-10:]})
                    salbatu = None
                    yes=0
                    continue
            try:
                if salbatu is None:
                    continue
                erref=json.loads(salbatu)
            except: #Detect unexpected formats from the model
                erref=None
                log.append({"Sentence": en, 
                            "Error": "Serialization error: "+str(salbatu)})
            erantzunak, log, bikoiztua = eguneratu_emaitza(erantzunak,max_ind,erref,log,en)
            if max_ind != len(erantzunak) and not bikoiztua:
                log.append({"Sentence": en,
                            "Error": "Order error: answer "+str(max_ind)+" is in the "+str(en)+" position"})
        error=False
    else:
        error=True
    erantzunak=sorted(erantzunak, key=lambda d: d['answer_id'])
    return erantzunak, log, error

#Postprocessing starter
def txukundu_emaitzak (emaitzak,berez_kop,log):
    if len(emaitzak) != berez_kop:
        daudenak = [x["answer_id"] for x in emaitzak]
        for zenb in range(1,berez_kop+1):
            if zenb not in daudenak:
                emaitzak.append({"answer_id": zenb, "evidence_id": None})
                log.append({"Sentence": zenb,
                            "Error": "Answer sentence "+str(zenb)+ " is not in the provided results"})
        emaitzak=sorted(emaitzak, key=lambda d: d['answer_id'])
    return emaitzak, log

#Main function to postprocess the generations of the model
def postproz (iragarpena,berezkoa):
    berezkoak=[]
    ber=berezkoa.split("\n")
    for i in ber:
        if len(i) == 0:
            continue
        berezkoak.append(i.split("[")[0])
    erantzunak, log, error = esperotako_formatua(iragarpena,berezkoak,"\n")
    dif1=len(berezkoak)-len(erantzunak)
    if error or dif1 != 0:
        erantzunak, log, error = esperotako_formatua(iragarpena,berezkoak,"]")
        dif2=len(berezkoak)-len(erantzunak)
        if not error and dif2 < dif1:
            log.append({"Sentence": 0,
                        "Error": "The sentences are horizontally presented"})
        erantzunak, log=txukundu_emaitzak(erantzunak,len(berezkoak),log)
        return erantzunak, log
    elif dif1 == 0:
        return erantzunak, log
    else:
        log.append({"Sentence": 0,
                "Error": "Bad format: the general format of the answer is far from the requested"})
    return None, log

##########################################################################################

        #VOTATION FUNCTIONS: Functions to combine the answers generated by the models 
        #using the majority voting rule. 

##########################################################################################

#Function to extract the reference numbers of the answers generated by the models and to proceed with the votation.
def votation (prozak):
    botoak = { }
    azken_emaitza = [ ]
    iters=len(prozak)
    noneak={}
    for en, proz in enumerate(prozak):
        if proz is None:
            continue
        if en == 0:
            for galdera in proz:
                if galdera["evidence_id"] is None:
                    botoak[galdera["answer_id"]]= None
                    noneak[galdera["answer_id"]]=1
                else:
                    noneak[galdera["answer_id"]]=0
                    botoak[galdera["answer_id"]]={}
                    for ida in galdera["evidence_id"]:
                        botoak[galdera["answer_id"]][ida]=1
        else:
            for galdera in proz:
                if galdera["evidence_id"] is None:
                    noneak[galdera["answer_id"]]+=1
                elif len(galdera["evidence_id"]) == 0:
                    botoak[galdera["answer_id"]]={}
                else:
                    for ida in galdera["evidence_id"]:
                        if botoak[galdera["answer_id"]] is None:
                            botoak[galdera["answer_id"]]={}
                        if ida not in botoak[galdera["answer_id"]]:
                            botoak[galdera["answer_id"]][ida]=1
                        else:
                            botoak[galdera["answer_id"]][ida]+=1                      
    for galdera in botoak:
        galdera_emaitza={"answer_id": galdera, "evidence_id": [ ]}
        if botoak[galdera] is None:
            galdera_emaitza["evidence_id"]=None
        else:
            for botoa in botoak[galdera]:
                kopurua=botoak[galdera][botoa]
                if kopurua > (iters-noneak[galdera])/ (1/MAJORITY):
                    galdera_emaitza["evidence_id"].append(botoa)
        azken_emaitza.append(galdera_emaitza)
    return azken_emaitza

#Main function for the votation
def iterations_voting (iter_results,berezkoa):
    prozak = [ ]
    logak = [ ]
    for emaitza in iter_results:
        emaitza=emaitza.split("Your answer:")[-1]
        proz, log=postproz(emaitza,berezkoa)
        prozak.append(proz)
        logak.append(log)
    boto_ostean=votation(prozak)
    return boto_ostean

##########################################################################################

        #FUNCTIONS TO PREPROCESS THE DATA OF THE SHARED TASK: To accommodate them for
        #the models. 

##########################################################################################

#Adapt the clinical notes
def galdera_formatu_emailea (galdera_erref,kasua):
    formatatuta=""
    for esaldia in galdera_erref[kasua]:
        formatatuta=formatatuta+esaldia+". -> "+galdera_erref[kasua][esaldia]+"\n"
    return formatatuta

#Adapt the clinical answers
def erantzun_formatu_emailea (erantzun_erref,kasua):
    formatatuta=""
    for esaldia in erantzun_erref[kasua]:
        formatatuta=formatatuta+"* "+erantzun_erref[kasua][esaldia]["Testua"]+" [??]"+"\n"
    return formatatuta

#Initialize GPUs
def gpu_initializer ():
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

#Initialize models
def model_initializer (device, mn, floata):
    if floata == 16:
        dtype=torch.float16
    if floata == 32:
        dtype=torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(mn)
    model = AutoModelForCausalLM.from_pretrained(
        mn,
        dtype=dtype,
        device_map="auto",# Use float16 for memory efficiency
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MODEL_PARAMS["tokens"],  # Maximum tokens to generate
        temperature=MODEL_PARAMS["temperature"],  # Controls randomness (0=deterministic, 1=creative)
        do_sample=MODEL_PARAMS["do_sample"]  # Enable sampling for more diverse outputs
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

#Initialize prompts
def prompt_loader ():
    with open(PROMPT_FILE1,"r") as f:
        prompt1 = f.read()
    with open(PROMPT_FILE2,"r") as f:
        prompt2 = f.read()
    return prompt1, prompt2

#Load preprocessed files
def preproc_files_loader ():
    with open(PREPROC_FILES["clinical_sentences"],"r") as f:
        clinical_sentences = json.load(f)
    with open(PREPROC_FILES["patient_question"],"r") as f:
        patient_question = json.load(f)
    with open(PREPROC_FILES["answer_sentences"],"r") as f:
        answer_sentences = json.load(f)
    return clinical_sentences, patient_question, answer_sentences

#Execute models
def model_executor (llm1,llm2,template1,template2,clinical_sentences, patient_question, answer_sentences, ITERS1,ITERS2):
    prompt1 = PromptTemplate(
        input_variables=PROMPT_VARIABLES,
        template=template1
    )
    prompt2 = PromptTemplate(
        input_variables=PROMPT_VARIABLES,
        template=template2
    )

    assert(len(clinical_sentences) == len(patient_question))
    assert(len(clinical_sentences) == len(answer_sentences))

    emaitzak = [ ]
    denborak = []
    for kasua in tqdm(clinical_sentences):
        iter_results = [ ]

        formated1 = prompt1.format(
                patient_question= patient_question[kasua],
                answer_sentences= erantzun_formatu_emailea(answer_sentences,kasua),
                clinical_sentences= galdera_formatu_emailea(clinical_sentences,kasua)
        )
        formated2 = prompt2.format(
                patient_question= patient_question[kasua],
                answer_sentences= erantzun_formatu_emailea(answer_sentences,kasua),
                clinical_sentences= galdera_formatu_emailea(clinical_sentences,kasua)
        )
        start = time.time()
        generazioak1 = llm1.generate([formated1] * int(ITERS1))
        generazioak2 = llm2.generate([formated2] * int(ITERS2))
        end = time.time()
        torch.cuda.empty_cache()
        gc.collect()
        for iterr in range(len(generazioak1.generations)):
            iter_results.append(generazioak1.generations[iterr][0].text)
        for iterr in range(len(generazioak2.generations)):
            iter_results.append(generazioak2.generations[iterr][0].text)
            #iter_results.append(generazioak["text"])
        if ITERS1 + ITERS2 > 1:
            result = iterations_voting(iter_results,erantzun_formatu_emailea(answer_sentences,kasua))
        else:
            emaitza=iter_results[0].split("Your answer:")[-1]
            result, log=postproz(emaitza,erantzun_formatu_emailea(answer_sentences,kasua))    
        emaitzak.append({"case_id": kasua, "prediction": result})
        denborak.append({"case_id": kasua, "time": end-start})
    return emaitzak, denborak

#Function to perform a dynamic evaluation
def finish_evaluation (gold,emaitzak,denborak):
    log_final = [ ]
    for j in range(MULTIPLE):
        ebal=ebaluatzailea(gold,emaitzak[j])
        micro=micro_ebal(ebal)
        log = prepare_results(ebal,micro)
        log["Exec_time"]=sum(pd.DataFrame(denborak[j])["time"])
        log_final.append(log)
    df = pd.DataFrame(log_final)
    mean=dict(df[["Precision","Recall","F1","Full_ok","Big_ok","Void_ok","Case_ok","Error","TP","Sent_error","Exec_time"]].mean())
    mean = {k: float(v) for k, v in mean.items()}
    std=dict(df[["Precision","Recall","F1","Full_ok","Big_ok","Void_ok","Case_ok","Error","TP","Sent_error","Exec_time"]].std())
    std = {k: float(v) for k, v in std.items()}
    return mean, std

#Main function
def main ():
    print("--CROSS-CONSISTENCY PROGRAM--")
    template1, template2 = prompt_loader ()
    print("Prompta topatuta.")
    clinical_sentences, patient_question, answer_sentences = preproc_files_loader()
    with open(GOLD_PATH,"r") as f:
        gold=json.load(f)
    print("Preprocessed files: FOUND.")

    device = gpu_initializer ()
    print("GPU: Initialized.")
    if FLOAT_POINT == "normal":
        llm1 = model_initializer (device, MODEL_NAME1,16)
    else:
        llm1 = model_initializer (device, MODEL_NAME1,32)
    if FLOAT_POINT == "normal":
        llm2 = model_initializer (device, MODEL_NAME2,16)
    else:
        llm2 = model_initializer (device, MODEL_NAME2,32)
    print("MODELS: Initialized.")

    results_col=[]
    times_col=[]
    for iter in tqdm(range(MULTIPLE)):
        results, times = model_executor(llm1,llm2,template1,template2,clinical_sentences,patient_question,answer_sentences,ITERS1,ITERS2)
        results_col.append(results)
        times_col.append(times)
    print("Exekuzio arrakastatsua.")
    with open(OUTPUT_FILE,"w") as f:
        json.dump(results_col,f)
    with open(TIMES_FILE,"w") as f:
        json.dump(times_col,f)
    #mean, std = finish_evaluation(gold,results_col,times_col)
    #print("Mean: ",mean)
    #print("Std: ",std)
    print("SUCCESS. Results saved.")

if __name__=="__main__":
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    else:
        print("No CUDA GPU available")
    main()
