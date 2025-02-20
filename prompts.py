ground_prompt = """Output the location of target element according to the given screenshot and instruction.
"""

box2func = """Output the functionality of target element according to the given screenshot and box.
"""

box2func_with_som = """Output the functionality of target element according to the given screenshot and box. Note the target element is hightlighted by red box.
"""

box2func_with_ocr = """Output the functionality of target element according to the given screenshot, box and OCR result.
"""

box2func_with_ocr_and_som = """Output the functionality of target element according to the given screenshot, box and OCR result. Note the target element is hightlighted by red box.
"""

ground_prompt_test = """Output the location of target element according to the given screenshot and instruction.
Instruction: {instruction}"""


box2func_test = """Output the functionality of target element according to the given screenshot and box.
Box: ({x1},{y1}),({x2},{y2})"""

box2func_with_som_test = """Output the functionality of target element according to the given screenshot and box. Note the target element is hightlighted by red box.
Box: ({x1},{y1}),({x2},{y2})"""


box2func_with_ocr_test = """Output the functionality of target element according to the given screenshot, box and OCR result.
Box: ({x1},{y1}),({x2},{y2})
Text: {text}"""

box2func_with_ocr_and_som_test = """Output the functionality of target element according to the given screenshot, box and OCR result. Note the target element is hightlighted by red box.
Box: ({x1},{y1}),({x2},{y2})
Text: {text}"""



agent_prompt = """Please generate the next move according to the given screenshot, instruction and previous actions. 

## Instruction: 
{instruction}

## Previous actions: 
{action_history}
"""


agent_action_caption_aitw = """Based on the provided screenshot, user instruction, and historical information, infer the intention of user's current action.
## Instruction:
{instruction}

## Previous actions:
{action_history}

The information about the user's current action is as follows
## Current action:
type: {action_type}, value: {action_value}
"""

agent_action_caption_mind2web = """Based on the provided screenshot, user instruction, and historical information, infer the intention of user's current action.
## Instruction:
{instruction}

## Previous actions:
{action_history}

The information about the user's current action is as follows
## Current action:
type: {action_type}, point: {point}, text: {text}
"""



all_prompts = dict(
    ground_prompt=ground_prompt,
    box2func=box2func,
    box2func_with_som=box2func_with_som,
    box2func_with_ocr=box2func_with_ocr,
    box2func_with_ocr_and_som=box2func_with_ocr_and_som,
    ground_prompt_test=ground_prompt_test,
    box2func_test=box2func_test,
    box2func_with_som_test=box2func_with_som_test,
    box2func_with_ocr_test=box2func_with_ocr_test,
    box2func_with_ocr_and_som_test=box2func_with_ocr_and_som_test,
    agent_action_caption_aitw=agent_action_caption_aitw,
    agent_prompt=agent_prompt,
    agent_action_caption_mind2web=agent_action_caption_mind2web
)
