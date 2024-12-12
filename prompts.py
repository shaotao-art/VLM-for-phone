ground_prompt_for_train = """<image>Output the location of target element according to the given instruction.
## Instruction
{instruction}"""

agent_prompt_for_train = """<image>Please generate the next move according to the given screenshot, instruction and previous actions. 

## Instruction: 
{instruction}

## Previous actions: 
{action_history}
"""


all_prompts = {'ground_prompt_for_train': ground_prompt_for_train,
               'agent_prompt_for_train': agent_prompt_for_train,
               'ground_prompt_for_val': ground_prompt_for_train.replace('<image>', ''),
               'agent_prompt_for_val': agent_prompt_for_train.replace('<image>', '')}
