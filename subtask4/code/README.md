# Instructions to execute the programs of Substak 4
The exact commands to execute the programs and to obtain the results described in the paper are described in the following sections. 

## Random and embedding models

Just execute the random_and_embedding_systems.py file.

## Zero-shot models

Execute the following command:

```
python3 zero_shot.py -pf $PROMPT_FILE$ -of $OUTPUT_FILE$ -tf $TIMES_FILE$ -mo $MODEL_NAME$ -fp $QUANTIZATION$
```

Where: 

PROMPT_FILE is the path to the file containing the prompt, in raw text.\n
OUTPUT_FILE is the path where the outputs should be placed.
TIMES_FILE is the path where the execution time numbers should be placed.
MODEL_NAME is the name of the LLM that will be executed. Identification number for loading via the transformers package.
QUANTIZATION has two options: "normal" to use float16 and "gemma" to use bfloat16. Check the requirements of the model. 

## Self-consistency models

Execute the following command:

```
python3 self_consistency.py -pf $PROMPT_FILE$ -of $OUTPUT_FILE$ -tf $TIMES_FILE$ -mo $MODEL_NAME$ -fp $QUANTIZATION$ -it $NUM_AGENTS$ -maj $MAJORITY_THRESHOLD$
```

Where: 

(besides the parameters explained in the previous section)
NUM_AGENTS is the number of instances of the LLM that will be executed to participate in the voting. 
MAJORITY_THRESHOLD is the minimum proportion of occurrences of a citation to be included in the final result. Range: 0 to 1. For example: 0.25 means that if the citation is provided by the 25% of the LLMs, it will be included in the final answer.  


## Cross-consistency models

Execute the following command:

```
python3 cross_consistency.py -pf1 $PROMPT_FILE1$ -pf2 $PROMPT_FILE2$ -of $OUTPUT_FILE$ -tf $TIMES_FILE$ -mo1 $MODEL_NAME1$ -mo2 $MODEL_NAME2$ -fp $QUANTIZATION$ -it1 $NUM_AGENTS1$ -it2 $NUM_AGENTS2$ -maj $MAJORITY_THRESHOLD$
```

Where: 

(besides the parameters explained in the previous section)
PROMPT_FILE2 is the path to the prompt file that the second LLM will use.
MODEL_NAME2 is the name of the second LLM.
NUM_AGENTS2 is the number of instances of the second LLM that will be executed. 
