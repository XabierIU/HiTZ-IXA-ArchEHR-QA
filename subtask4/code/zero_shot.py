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
import argparse

parser = argparse.ArgumentParser(description='Simple agent caller function.')

parser.add_argument('-pf', '--prompt', help='Path to prompt file')
parser.add_argument('-of', '--output', help='Path to output file')
parser.add_argument('-tf', '--time', help='Path to times file')
parser.add_argument('-mo', '--model', help='Model name')
parser.add_argument('-fp', '--float', help='Model name')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

PREPROC_FILES = {"clinical_sentences": "../preproc/TEST-2025/ohar_klinikoa.json",
                 "patient_question": "../preproc/TEST-2025/galderak.json",
                 "answer_sentences": "../preproc/TEST-2025/erantzunak.json"}

PROMPT_FILE = args.prompt #"./prompts/prompt_gabe.txt"
PROMPT_VARIABLES = ["clinical_sentences",
                    "patient_question", 
                    "answer_sentences"]

FLOAT_POINT = args.float

GPU_ID = 0

MODEL_NAME = args.model #"HPAI-BSC/Llama3.1-Aloe-Beta-8B"
#MODEL_NAME = "ibm-granite/granite-4.0-h-tiny"
#MODEL_NAME = "meta-llama/Llama-3.1-8B"
#MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"
#MODEL_NAME = "Qwen/Qwen3-8B"
#MODEL_NAME = "johnsnowlabs/JSL-MedLlama-3-8B-v2.0"
#MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
#MODEL_NAME = "google/medgemma-4b-it"
#MODEL_NAME = "google/medgemma-27b-it"

MODEL_PARAMS = {"tokens": 256, 
                "temperature": 0.7, 
                "do_sample": True}

OUTPUT_FILE = args.output #"../results/ALOE/PROMPT-EGITURATUA/outputs.json"
TIMES_FILE = args.time #"../results/ALOE/PROMPT-EGITURATUA/times.json"

ITERS = 1

torch.cuda.empty_cache()

#Function to give a structured format to the clinical notes.
def galdera_formatu_emailea (galdera_erref,kasua):
    formatatuta=""
    for esaldia in galdera_erref[kasua]:
        formatatuta=formatatuta+esaldia+". -> "+galdera_erref[kasua][esaldia]+"\n"
    return formatatuta

#Function to give a structured format to the clinical answer.
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
def model_initializer (device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if FLOAT_POINT == "normal":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,  # Use float16 for memory efficiency
        ).to(device)
    elif FLOAT_POINT == "gemma":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,  # Use float16 for memory efficiency
        ).to(device)

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
    with open(PROMPT_FILE,"r") as f:
        prompt = f.read()
    return prompt

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
def model_executor (llm,template,clinical_sentences, patient_question, answer_sentences):
    prompt = PromptTemplate(
        input_variables=PROMPT_VARIABLES,
        template=template
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    assert(len(clinical_sentences) == len(patient_question))
    assert(len(clinical_sentences) == len(answer_sentences))

    emaitzak = [ ]
    denborak = []
    for kasua in tqdm(clinical_sentences):
        start = time.time()
        result = chain.invoke({
            "patient_question": patient_question[kasua],
            "answer_sentences": erantzun_formatu_emailea(answer_sentences,kasua),
            "clinical_sentences": galdera_formatu_emailea(clinical_sentences,kasua)
        })
        end = time.time()
        torch.cuda.empty_cache()
        gc.collect()
        emaitzak.append(result["text"])
        denborak.append({"case_id": kasua, "time": end-start})
    
    return emaitzak, denborak

#Main function
def main ():
    print("--SIMPLE AGENT PROGRAM--")
    template = prompt_loader ()
    print("PROMPT: FOUND.")
    clinical_sentences, patient_question, answer_sentences = preproc_files_loader()
    print("Fitxategi aurreprozesatuak topatuta.")

    device = gpu_initializer ()
    print("GPU: Initialized.")
    llm = model_initializer (device)
    print("MODEL: Initialized.")
    if ITERS == 0:
        results, times = model_executor(llm,template,clinical_sentences,patient_question,answer_sentences)
        print("EXECUTION: SUCCESS.")
        with open(OUTPUT_FILE,"w") as f:
            json.dump(results,f)
        with open(TIMES_FILE,"w") as f:
            json.dump(times,f)
    else:
        results_col=[]
        times_col=[]
        for iter in tqdm(range(ITERS)):
                results, times = model_executor(llm,template,clinical_sentences,patient_question,answer_sentences)
                results_col.append(results)
                times_col.append(times)
        print("EXECUTION: SUCCESS.")
        with open(OUTPUT_FILE,"w") as f:
            json.dump(results_col,f)
        with open(TIMES_FILE,"w") as f:
            json.dump(times_col,f)
    print("RESULTS: SAVED.")

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
