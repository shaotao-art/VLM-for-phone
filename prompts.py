ground_prompt = """Output the location of target element according to the given screenshot and instruction.
"""

box2func = """Output the functionality of target element according to the given screenshot and box.
"""

box2func_with_som = """Output the functionality of target element marked by a box according to the given screenshot and box.
"""

box2func_with_ocr = """Output the functionality of target element according to the given screenshot, box and OCR result.
"""

box2func_with_ocr_and_som = """Output the functionality of target element marked by a box according to the given screenshot, box and OCR result.
"""

agent_prompt_for_train = """Please generate the next move according to the given screenshot, instruction and previous actions. 

## Instruction: 
{instruction}

## Previous actions: 
{action_history}
"""


short_answer_template = """{{"Function": {function}}}"""
long_answer_template = """{{"Oberservation": {observation},  "Thoughts": {thoughts}, "Action": {action}, "Function": {function}}}"""

agent_multilvel_annotation_prompt = """Please generate the anly process according to the given screenshot, instruction, previous actions, gt element's infomation and GT action. target element's box is marked by a box in the screenshot and its bounding box is given below.

## Instruction:
{instruction}

## Previous actions:
{action_history}

## Ground truth elements Box:
({x1},{y1}),({x2},{y2})

## GT action:
type: {action_type}, value: {action_value}

You need to generate the process to achieve the GT action on the GT element, your output should follow the format below:
{{"Oberservation": "<The current state of the screen>",  "Thoughts": "<thinking process to complete the task according to the observation>", "action": "<the action you need to take to complete the task based on your anlysies>"}}
"""



all_prompts = dict(
    ground_prompt=ground_prompt,
    box2func=box2func,
    box2func_with_som=box2func_with_som,
    box2func_with_ocr=box2func_with_ocr,
    box2func_with_ocr_and_som=box2func_with_ocr_and_som,
    agent_prompt_for_train=agent_prompt_for_train
)
