"""
HumanEval SCoT Code Generation - Complete Working Version
SCoT → Code → Run → Fix if errors → Evaluation
"""

import json
import os
import subprocess
import sys
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from human_eval.data import read_problems

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate

# ==================== Configuration ====================
@dataclass
class Config:
    model_name: str = "gpt-4o"
    scot_temperature: float = 0.8
    code_temperature: float = 0.0
    max_tokens: int = 2048
    api_key: str = None

# ==================== Prompt Templates ====================
SCOT_GENERATION_PROMPT = """You are an expert programmer. Generate a structured chain-of-thought (SCoT) for solving the following problem.

Use these program structures in your reasoning:
1. **Sequence Structure**: Break down the problem into sequential steps
2. **Branch Structure**: Identify conditional logic (if-else statements)
3. **Loop Structure**: Determine iterative processes needed
4. **Input-Output Structure**: Define clear input parameters and expected output

Format your SCoT as follows:
- Start with understanding the problem requirements
- List the algorithmic steps using the structures above
- Consider edge cases and error handling
- End with the return statement specification

Problem Description:
{problem_description}

Let's think step by step.

Structured Chain-of-Thought:"""

CODE_GENERATION_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable Python code.

Given the problem description and structured chain-of-thought (SCoT), implement the solution.
IMPORTANT: 
- Double-check the SCoT and fix any logical issues
- Include comprehensive docstrings with parameter types and return types
- Handle edge cases explicitly
- Ensure the code is self-contained (no undefined variables)
- Add input validation where necessary

Problem Description:
{problem_description}

Structured Chain-of-Thought:
{scot}

Now implement the complete Python solution:

```python
{function_signature}
    \"\"\"
    {docstring}
    \"\"\"
"""

ERROR_CORRECTION_PROMPT = """The following code produced an error when tested:

Code:
```python
{code}
```

Error:
{error}

Problem Description:
{problem_description}

Please fix the code to handle this error properly. Focus on:
1. Input validation
2. Edge case handling
3. Type consistency
4. Variable scope issues

Fixed code:
```python"""

# ==================== Sample Problems ====================
def get_sample_problems() -> List[Dict]:
    return list(read_problems().values())

# ==================== Helper Functions ====================
def extract_code(response: str) -> str:
    """Extract Python code from model response"""
    if "```python" in response:
        match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1)
    elif "```" in response:
        match = re.search(r'```\n(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1)
    return response.strip()

def parse_problem(prompt: str) -> Tuple[str, str]:
    """Extract function signature and docstring from problem prompt"""
    lines = prompt.strip().split('\n')
    func_sig = ""
    docstring = ""
    
    for i, line in enumerate(lines):
        if line.strip().startswith('def '):
            func_sig = line
            # Extract docstring
            for j in range(i+1, len(lines)):
                if '"""' in lines[j]:
                    for k in range(j+1, len(lines)):
                        if '"""' in lines[k]:
                            docstring = '\n'.join(lines[j:k+1])
                            break
                    break
            break
    
    return func_sig, docstring

def post_process_code(code: str, entry_point: str) -> str:
    """Clean up generated code"""
    lines = [line for line in code.split('\n') 
             if not (line.strip().startswith('from typing import') or 
                    (line.strip().startswith('import') and 'typing' in line))]
    
    code = '\n'.join(lines)
    
    # Ensure function name matches entry point
    if f"def {entry_point}" not in code:
        code = re.sub(r'def \w+\(', f'def {entry_point}(', code, count=1)
    
    return code

def run_code_with_tests(code: str, test_code: str, entry_point: str) -> Tuple[bool, str]:
    """Run code with test cases to check if it works"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        full_code = f"""
from typing import List, Dict, Any, Optional, Tuple, Union

{code}

{test_code}

check({entry_point})
"""
        f.write(full_code)
        f.flush()
        temp_file = f.name
    
    try:
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        success = result.returncode == 0
        error_msg = "Success" if success else result.stderr
        return success, error_msg
        
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)

# ==================== Core Generation Functions ====================
def generate_scot(llm: ChatOpenAI, problem_description: str) -> str:
    """Generate Structured Chain-of-Thought reasoning"""
    print("→ Generating SCoT reasoning...")
    
    prompt = PromptTemplate(template=SCOT_GENERATION_PROMPT, input_variables=["problem_description"])
    messages = [
        SystemMessage(content="You are an expert programmer who excels at breaking down complex problems into structured reasoning steps."),
        HumanMessage(content=prompt.format(problem_description=problem_description))
    ]
    
    response = llm.invoke(messages)
    print("✓ SCoT generated")
    return response.content

def generate_code_from_scot(llm: ChatOpenAI, problem_description: str, scot: str, func_sig: str, docstring: str) -> str:
    """Generate code from SCoT"""
    print("→ Generating code from SCoT...")
    
    prompt = PromptTemplate(
        template=CODE_GENERATION_PROMPT,
        input_variables=["problem_description", "scot", "function_signature", "docstring"]
    )
    
    messages = [
        SystemMessage(content="You are a Python expert who writes clean, efficient, and bug-free code."),
        HumanMessage(content=prompt.format(
            problem_description=problem_description, 
            scot=scot, 
            function_signature=func_sig,
            docstring=docstring
        ))
    ]
    
    response = llm.invoke(messages)
    code = extract_code(response.content)
    print("✓ Code generated")
    return code

def fix_code_with_error(llm: ChatOpenAI, code: str, error: str, problem_description: str) -> str:
    """Fix code that produced an error"""
    print("→ Fixing code error...")
    
    prompt = PromptTemplate(
        template=ERROR_CORRECTION_PROMPT,
        input_variables=["code", "error", "problem_description"]
    )
    
    messages = [
        SystemMessage(content="You are an expert debugger who fixes Python code errors."),
        HumanMessage(content=prompt.format(code=code, error=error, problem_description=problem_description))
    ]
    
    response = llm.invoke(messages)
    fixed_code = extract_code(response.content)
    print("✓ Code fixed")
    return fixed_code

# ==================== Main Processing ====================
def process_single_problem(config: Config, problem: Dict) -> bool:
    """Process a single problem: SCoT → Code → Run → Fix if needed. Returns True if successful."""
    print(f"\n{'='*60}")
    print(f"Processing: {problem['task_id']}")
    print(f"{'='*60}")
    
    # Create LLM instances
    scot_llm = ChatOpenAI(
        model=config.model_name,
        temperature=config.scot_temperature,
        max_tokens=500,
        api_key=config.api_key
    )
    
    code_llm = ChatOpenAI(
        model=config.model_name,
        temperature=config.code_temperature,
        max_tokens=config.max_tokens,
        api_key=config.api_key
    )
    
    prompt_text = problem['prompt']
    func_sig, docstring = parse_problem(prompt_text)
    
    # Step 1: Generate SCoT
    scot = generate_scot(scot_llm, prompt_text)
    
    if not scot:
        print("✗ Failed to generate SCoT")
        return False
    
    # Step 2: Generate Code from SCoT
    code = generate_code_from_scot(code_llm, prompt_text, scot, func_sig, docstring)
    
    if not code:
        print("✗ Failed to generate code")
        return False
    
    # Step 3: Post-process code
    code = post_process_code(code, problem['entry_point'])
    
    # Step 4: Run code with tests
    print("→ Running code with tests...")
    success, error = run_code_with_tests(code, problem['test'], problem['entry_point'])
    
    if success:
        print("✓ Code passed all tests!")
        return True
    else:
        print(f"✗ Code failed: {error.split('Error:')[-1].strip()}")
        
        # Step 5: Fix the error
        fixed_code = fix_code_with_error(code_llm, code, error, prompt_text)
        fixed_code = post_process_code(fixed_code, problem['entry_point'])
        
        # Step 6: Test fixed code
        print("→ Testing fixed code...")
        success_fixed, error_fixed = run_code_with_tests(fixed_code, problem['test'], problem['entry_point'])
        
        if success_fixed:
            print("✓ Fixed code passed all tests!")
            return True
        else:
            print(f"✗ Fixed code still failed: {error_fixed.split('Error:')[-1].strip()}")
            return False

# ==================== Evaluation Function ====================
def evaluate_results(results: List[bool]) -> Dict[str, Any]:
    """Simple evaluation function that calculates Pass@1 rate"""
    total_problems = len(results)
    successful_problems = sum(results)
    pass_at_1 = successful_problems / total_problems if total_problems > 0 else 0.0
    
    evaluation = {
        'total_problems': total_problems,
        'successful_problems': successful_problems,
        'failed_problems': total_problems - successful_problems,
        'pass_at_1': pass_at_1,
        'pass_at_1_percentage': f"{pass_at_1 * 100:.2f}%"
    }
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total Problems: {evaluation['total_problems']}")
    print(f"Successful: {evaluation['successful_problems']}")
    print(f"Failed: {evaluation['failed_problems']}")
    print(f"Pass@1 Rate: {evaluation['pass_at_1_percentage']}")
    print(f"{'='*60}")
    
    return evaluation

def main():
    """Main function - processes all HumanEval problems with evaluation"""
    API_KEY = "your-openai-api-key"
    
    config = Config(api_key=API_KEY)
    problems = get_sample_problems()[:2]
    print(problems)
    
    print("Starting SCoT Code Generation Pipeline")
    print(f"Model: {config.model_name}")
    print(f"Total problems: {len(problems)}")
    
    # Track results for evaluation
    results = []
    
    # Process each problem one by one
    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}]", end=" ")
        success = process_single_problem(config, problem)
        results.append(success)
    
    # Evaluate final results
    evaluation = evaluate_results(results)
    
    # Save results to file
    with open('humaneval_evaluation.json', 'w') as f:
        json.dump(evaluation, f, indent=2)
    
    print(f"\nEvaluation saved to humaneval_evaluation.json")
    print(f"Pipeline complete!")

if __name__ == "__main__":
    main()
