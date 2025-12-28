convert_reasoning_prompt = """
{qa_block}

I want to convert this question into the style of a reasoning problem.
For example, here is such an example:

Question: A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there?
Candidates: (A) 1 (B) 4 (C) 5 (D) 10 (E) 14
Answer: (A) 4

In this case, please add intermediate reasoning steps and the candidate options at each step, like below:

Step 1:
- Reasoning: The juggler can juggle 16 balls.
- Candidates: A, B, C, D

Step 2:
- Reasoning: Half of the balls are golf balls.
- Candidates: A, B, C

Step 3:
- Reasoning: Now, there are 8 golf balls.
- Candidates: A, B, C

Step 4:
- Reasoning: Half of the golf balls are blue, and half of 8 is 4.
- Candidates: A

Final Answer: A

In this example:
- In Step 1, the number of balls does not narrow down the answer yet, so all candidates are kept.
- In Step 2, the fact that half of the balls are golf balls eliminates option D, so A, B, C remain.
- In Step 3, we know there are 8 golf balls, so A, B, C remain.
- In Step 4, we learn that half of the golf balls are blue, so the correct answer is A.
- THE IMPORTANT POINT IS THAT THROUGH THE REASONING STEPS, THE CANDIDATES SHOULD BE GRADUALLY NARROWED DOWN. PLEASE AVOID SITUATIONS WHERE THE CANDIDATES SUDDENLY DROP TO ONLY ONE OPTION.

Please convert the question into this format by adding such reasoning steps and the remaining candidate options.

The output format should be:

Step 1:
- Reasoning: ...
- Candidates: ...
Step 2:
- Reasoning: ...
- Candidates: ...
...


Cautions:
- Follow all formatting instructions strictly.
- For candidates, include only the option labels: A, B, C, D, or E. (Do not include brackets or any extra symbols.)
- Ensure that the reasoning gradually reduces the number of candidates step by step. For example, if there are 9 steps, avoid situations where the number of candidates suddenly drops to only one at Step 8 or Step 9.
- PLEASE ANSSER IN ONE TIME.
- In the final step of the reasoning, make sure to arrive at a definitive final answer. Do not end with multiple remaining candidates.
"""
