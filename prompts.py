ground_prompt_for_train = """<image>Output the location of target element according to the given instruction.
## Instruction
{instruction}"""

box2func_prompt_for_train = """<image>Please output the functionality of the target element according to the given box.
## Box
({x1},{y1}),({x2},{y2})"""

box2func_with_ocr_prompt_for_train = """<image>Please output the functionality of the target element according to the given box and ocr result.
## Box
({x1},{y1}),({x2},{y2})

## OCR result
{text}"""

agent_prompt_for_train = """<image>Please generate the next move according to the given screenshot, instruction and previous actions. 

## Instruction: 
{instruction}

## Previous actions: 
{action_history}
"""


all_prompts = {'ground_prompt_for_train': ground_prompt_for_train,
               'agent_prompt_for_train': agent_prompt_for_train,
               'box2func_prompt_for_train': box2func_prompt_for_train,
               'box2func_with_ocr_prompt_for_train': box2func_with_ocr_prompt_for_train,
               'ground_prompt_for_val': ground_prompt_for_train.replace('<image>', ''),
               'agent_prompt_for_val': agent_prompt_for_train.replace('<image>', ''),
               'box2func_prompt_for_val': box2func_prompt_for_train.replace('<image>', ''),
               'box2func_with_ocr_prompt_for_val': box2func_with_ocr_prompt_for_train.replace('<image>', '')}
