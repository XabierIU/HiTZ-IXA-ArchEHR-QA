# Instructions to execute the programs of Substak 2

To execute the programs used to reach the results outlined in the paper, it is sufficient to run the .py file in this folder. Inside the folder, some global variables must be adapted to your computer:

*  The paths to the original data files of the Shared Task.
* `MODEL_ID`: the name of the LLM that should be used.
* `ITERATIONS`: the global variable that sets the number of executions of the full program that should be done. Default value: 5 executions.

To produce the output without the use of the not-relevant finder agent and the coordinator agent, it is enough to comment out the caller lines of these agents in the coordinator function.
