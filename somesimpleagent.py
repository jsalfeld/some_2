"""
LangGraph Statistical Validation Test Agent

This agent performs statistical data analysis, validates results,
and generates comprehensive reports with full reasoning transparency.
"""

import os
from typing import TypedDict, Annotated, Literal
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import subprocess
import tempfile


# Define the state for our statistical analysis agent
class StatisticalAnalysisState(TypedDict):
    # Input
    task: str
    data_file_path: str
    output_dir: str  # Directory to save plots and output files

    # Analysis planning
    analysis_objective: str
    data_summary: str

    # Code generation & execution
    code: str
    execution_result: str
    execution_error: str

    # Report sections
    analysis_details: str
    analysis_conclusions: str

    # Reasoning/thought tracking
    reasoning_log: list

    # Control flow
    iteration: int
    max_iterations: int
    should_continue: bool
    is_valid: bool


# Initialize the OpenAI model
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)


# Helper function to log agent thoughts
def add_thought(state: StatisticalAnalysisState, node_name: str, thought: str) -> list:
    """Add a reasoning entry to the log."""
    reasoning_entry = {
        "node": node_name,
        "iteration": state.get("iteration", 0),
        "thought": thought,
        "timestamp": datetime.now().isoformat()
    }
    return state.get("reasoning_log", []) + [reasoning_entry]


def clean_code(code: str) -> str:
    """Remove markdown code fences from generated code."""
    code = code.strip()

    # Remove starting fence
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```py"):
        code = code[5:]
    elif code.startswith("```"):
        code = code[3:]

    # Remove ending fence
    if code.endswith("```"):
        code = code[:-3]

    code = code.strip()

    # Remove any remaining ``` lines (sometimes LLM adds them in the middle)
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are just markdown fences
        if stripped in ['```', '```python', '```py']:
            continue
        cleaned_lines.append(line)

    code = '\n'.join(cleaned_lines)

    return code.strip()


def understand_data(state: StatisticalAnalysisState) -> StatisticalAnalysisState:
    """First node: Examine the data file and generate a summary."""

    data_file_path = state["data_file_path"]
    output_dir = state.get("output_dir", ".")

    system_prompt = """You are a data analyst. Examine the provided data file and generate a comprehensive
    data summary. Your summary should include:
    - Data shape (rows, columns)
    - Column names and data types
    - Missing values
    - Basic descriptive statistics
    - Any data quality issues or notable patterns

    Provide a clear, concise summary that will help plan the statistical analysis."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""Please examine this data file: {data_file_path}
(The file is in the current working directory, so use this filename directly)

Generate a Python code snippet that loads the data and prints a comprehensive summary.
Only output the Python code without explanations or markdown formatting.""")
    ]

    response = llm.invoke(messages)
    code = clean_code(response.content)

    # Execute the code to get data summary
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        # Execute with output_dir as working directory so file can be found
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=output_dir  # Set working directory to find the data file
        )

        os.unlink(temp_file)
        data_summary = result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        data_summary = f"Error examining data: {str(e)}"

    # Log the reasoning
    thought = f"""Examined data file: {data_file_path}

Data Summary Generated:
{data_summary[:500]}...

This summary will guide our analysis approach."""

    reasoning_log = add_thought(state, "understand_data", thought)

    return {
        **state,
        "data_summary": data_summary,
        "reasoning_log": reasoning_log
    }


def plan_analysis(state: StatisticalAnalysisState) -> StatisticalAnalysisState:
    """Second node: Determine the analysis objective and approach."""

    task = state["task"]
    data_summary = state["data_summary"]

    system_prompt = """You are a statistical analyst. Based on the user's task and the data summary,
    determine:
    1. The clear analysis objective (what question are we answering?)
    2. The appropriate statistical methods/tests to use
    3. Any assumptions that need to be checked
    4. Expected outputs (tables, plots, test results)

    Be specific about the statistical approach and explain your reasoning."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""Task: {task}

Data Summary:
{data_summary}

Please provide:
1. A clear analysis objective statement
2. The recommended statistical approach with rationale""")
    ]

    response = llm.invoke(messages)
    analysis_plan = response.content.strip()

    # Extract objective (assume first paragraph or section)
    objective_lines = []
    for line in analysis_plan.split('\n'):
        if line.strip():
            objective_lines.append(line)
            if len(objective_lines) >= 3:  # Get first few meaningful lines as objective
                break

    analysis_objective = '\n'.join(objective_lines)

    # Log the reasoning
    thought = f"""Task Interpretation: {task}

Analysis Plan:
{analysis_plan}

This plan guides what statistical methods we'll implement."""

    reasoning_log = add_thought(state, "plan_analysis", thought)

    return {
        **state,
        "analysis_objective": analysis_objective,
        "reasoning_log": reasoning_log
    }


def write_analysis_code(state: StatisticalAnalysisState) -> StatisticalAnalysisState:
    """Third node: Generate Python code to perform the statistical analysis."""

    iteration = state.get("iteration", 0)
    task = state["task"]
    data_file_path = state["data_file_path"]
    output_dir = state.get("output_dir", ".")
    data_summary = state["data_summary"]
    analysis_objective = state["analysis_objective"]

    if iteration == 0:
        # First iteration - write initial code
        system_prompt = """You are an expert statistical programmer. Write Python code to perform statistical analysis.

Requirements:
- Load the data from the specified file path (file is in current working directory)
- Perform appropriate statistical tests and validation
- Check statistical assumptions (normality, homogeneity, independence, etc.)
- Generate informative visualizations and SAVE ALL PLOTS to current directory
- **CRITICAL**: Save plots with descriptive filenames (e.g., plt.savefig('distribution_plot.png'))
- DO NOT use plt.show() - only plt.savefig()
- Print structured results that include:
  * Test statistics and p-values
  * Confidence intervals where appropriate
  * Effect sizes
  * Assumption check results
- Handle missing data appropriately
- Use libraries like pandas, numpy, scipy, statsmodels, matplotlib, seaborn

Only output the Python code without explanations or markdown formatting.
The code should be production-ready and include error handling."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Task: {task}

Data File: {data_file_path}
(This file is in the current working directory, so just use this filename directly)

Data Summary:
{data_summary}

Analysis Objective:
{analysis_objective}

Write complete Python code to perform this analysis.
IMPORTANT: Save all plots to the current directory with descriptive names like 'var_distribution.png', 'rolling_var.png', etc.""")
        ]
    else:
        # Subsequent iterations - improve based on validation feedback
        system_prompt = """You are an expert statistical programmer. Based on the validation feedback,
improve the analysis code. Fix any errors, address failed assumptions, or enhance the analysis.

Only output the Python code without explanations or markdown formatting."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Task: {task}

Previous Code:
{state['code']}

Execution Result:
{state['execution_result']}

Execution Error (if any):
{state['execution_error']}

Validation Feedback:
{state.get('analysis_details', 'No specific feedback yet')}

Please write improved code based on this feedback.""")
        ]

    response = llm.invoke(messages)
    code = clean_code(response.content)

    # Log the reasoning
    if iteration == 0:
        thought = f"""Generated initial analysis code.

Approach:
- Loading data from {data_file_path}
- Implementing statistical tests based on the analysis plan
- Including assumption checks and validation
- Creating visualizations for results

Code length: {len(code)} characters
Libraries used: pandas, numpy, scipy, matplotlib (inferred from task)"""
    else:
        thought = f"""Iteration {iteration}: Revised analysis code.

Improvements made:
- Addressed execution errors or validation feedback
- Enhanced statistical rigor
- Improved error handling

Code length: {len(code)} characters"""

    reasoning_log = add_thought(state, "write_analysis_code", thought)

    return {
        **state,
        "code": code,
        "iteration": iteration + 1,
        "reasoning_log": reasoning_log
    }


def execute_code(state: StatisticalAnalysisState) -> StatisticalAnalysisState:
    """Fourth node: Execute the generated analysis code and capture the results."""

    code = state["code"]
    output_dir = state.get("output_dir", ".")

    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Create a temporary file to execute the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        # Execute the code with the output directory as the working directory
        # This ensures relative paths work and plots are saved in the right place
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=60,  # Increased timeout for statistical computations
            cwd=output_dir  # Set working directory to output directory
        )

        # Clean up temp file
        os.unlink(temp_file)

        execution_result = result.stdout
        execution_error = result.stderr if result.returncode != 0 else ""

        # Log the reasoning
        if execution_error:
            thought = f"""Code execution failed.

Error:
{execution_error[:300]}...

Will need to revise the code to fix these issues."""
        else:
            thought = f"""Code executed successfully.

Output preview:
{execution_result[:500]}...

Analysis produced results. Now validating statistical assumptions and completeness."""

        reasoning_log = add_thought(state, "execute_code", thought)

        return {
            **state,
            "execution_result": execution_result,
            "execution_error": execution_error,
            "reasoning_log": reasoning_log
        }

    except subprocess.TimeoutExpired:
        thought = "Code execution timed out after 60 seconds. Analysis may be too computationally intensive or contains infinite loops."
        reasoning_log = add_thought(state, "execute_code", thought)
        return {
            **state,
            "execution_result": "",
            "execution_error": "Code execution timed out after 60 seconds",
            "reasoning_log": reasoning_log
        }
    except Exception as e:
        thought = f"Unexpected error during execution: {str(e)}"
        reasoning_log = add_thought(state, "execute_code", thought)
        return {
            **state,
            "execution_result": "",
            "execution_error": f"Error during execution: {str(e)}",
            "reasoning_log": reasoning_log
        }


def validate_results(state: StatisticalAnalysisState) -> StatisticalAnalysisState:
    """Fifth node: Validate the analysis results and determine if improvements are needed."""

    system_prompt = """You are a statistical validator. Review the analysis code and results to determine:

1. EXECUTION STATUS:
   - Did the code execute successfully?
   - Are there any errors that need fixing?

2. STATISTICAL VALIDITY:
   - Were appropriate statistical tests used?
   - Were assumptions checked (normality, independence, homogeneity, etc.)?
   - Are the assumptions met? If not, were appropriate alternatives used?
   - Are effect sizes and confidence intervals reported?

3. COMPLETENESS:
   - Does the analysis fully address the task?
   - Are results interpretable and clearly presented?
   - Are visualizations appropriate?

4. ANALYSIS DETAILS:
   Provide a summary of:
   - Methods used
   - Assumptions checked and their results
   - Any transformations or adjustments made

5. ANALYSIS CONCLUSIONS:
   Provide:
   - Key findings with statistical support
   - Practical interpretation
   - Limitations
   - Recommendations (if applicable)

If improvements are needed, specify what should be changed. If the analysis is valid and complete, say "Analysis is valid and complete."
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""Task: {state['task']}

Code:
{state['code']}

Execution Result:
{state['execution_result']}

Execution Error (if any):
{state['execution_error']}

Please provide your validation assessment.""")
    ]

    response = llm.invoke(messages)
    validation = response.content.strip()

    # Parse the validation response to extract analysis details and conclusions
    # Look for sections in the response
    lines = validation.split('\n')
    analysis_details = []
    analysis_conclusions = []
    current_section = None

    for line in lines:
        line_lower = line.lower()
        if 'analysis details' in line_lower or 'methods used' in line_lower:
            current_section = 'details'
        elif 'analysis conclusions' in line_lower or 'key findings' in line_lower or 'conclusions' in line_lower:
            current_section = 'conclusions'
        elif current_section == 'details':
            analysis_details.append(line)
        elif current_section == 'conclusions':
            analysis_conclusions.append(line)

    # If sections weren't clearly identified, use a simpler heuristic
    if not analysis_details and not analysis_conclusions:
        midpoint = len(lines) // 2
        analysis_details = lines[:midpoint]
        analysis_conclusions = lines[midpoint:]

    analysis_details_str = '\n'.join(analysis_details).strip()
    analysis_conclusions_str = '\n'.join(analysis_conclusions).strip()

    # Determine if analysis is valid and complete
    is_valid = (
        "valid and complete" in validation.lower() or
        "no improvements needed" in validation.lower()
    ) and state["execution_error"] == ""

    # Determine if we should continue iterating
    should_continue = (
        state["iteration"] < state["max_iterations"] and
        not is_valid
    )

    # If there's an error, we should try to fix it
    if state["execution_error"] and state["iteration"] < state["max_iterations"]:
        should_continue = True

    # Log the reasoning
    thought = f"""Validation Assessment:

Execution Status: {"Success" if not state["execution_error"] else "Failed"}
Statistical Validity: {"Valid" if is_valid else "Needs improvement"}
Should Continue: {should_continue}

Validation Details:
{validation[:500]}...

{"Analysis meets all criteria and is ready for reporting." if is_valid else "Further iteration needed to address issues."}"""

    reasoning_log = add_thought(state, "validate_results", thought)

    return {
        **state,
        "analysis_details": analysis_details_str,
        "analysis_conclusions": analysis_conclusions_str,
        "is_valid": is_valid,
        "should_continue": should_continue,
        "reasoning_log": reasoning_log
    }


def generate_report(state: StatisticalAnalysisState) -> StatisticalAnalysisState:
    """Sixth node: Generate the final report and reasoning files."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    reasoning_log = state.get("reasoning_log", [])

    # ========================================================================
    # 1. Build the ANALYSIS REPORT (3 sections only - no reasoning)
    # ========================================================================
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("STATISTICAL ANALYSIS REPORT")
    report_lines.append(f"Generated: {timestamp}")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Section 1: Analysis Objective
    report_lines.append("SECTION 1: ANALYSIS OBJECTIVE")
    report_lines.append("-" * 80)
    report_lines.append(state.get("analysis_objective", "Not specified"))
    report_lines.append("")

    # Section 2: Analysis Details
    report_lines.append("SECTION 2: ANALYSIS DETAILS")
    report_lines.append("-" * 80)
    report_lines.append(state.get("analysis_details", "No details available"))
    report_lines.append("")

    # Section 3: Analysis Conclusions
    report_lines.append("SECTION 3: ANALYSIS CONCLUSIONS")
    report_lines.append("-" * 80)
    report_lines.append(state.get("analysis_conclusions", "No conclusions available"))
    report_lines.append("")

    report_lines.append("=" * 80)
    report_content = '\n'.join(report_lines)

    # ========================================================================
    # 2. Build the AGENT REASONING file (separate)
    # ========================================================================
    reasoning_lines = []
    reasoning_lines.append("=" * 80)
    reasoning_lines.append("AGENT REASONING PROCESS")
    reasoning_lines.append(f"Generated: {timestamp}")
    reasoning_lines.append("=" * 80)
    reasoning_lines.append("")
    reasoning_lines.append("This file contains the complete thought process of the statistical")
    reasoning_lines.append("analysis agent as it worked through the analysis task.")
    reasoning_lines.append("")
    reasoning_lines.append("=" * 80)
    reasoning_lines.append("")

    # Group by iteration
    iterations = {}
    for entry in reasoning_log:
        iter_num = entry.get("iteration", 0)
        if iter_num not in iterations:
            iterations[iter_num] = []
        iterations[iter_num].append(entry)

    for iter_num in sorted(iterations.keys()):
        if iter_num == 0:
            reasoning_lines.append("INITIAL ANALYSIS (Iteration 0):")
        else:
            reasoning_lines.append(f"\nITERATION {iter_num}:")
        reasoning_lines.append("-" * 80)

        for entry in iterations[iter_num]:
            node_name = entry.get("node", "unknown")
            thought = entry.get("thought", "")
            timestamp_entry = entry.get("timestamp", "")

            reasoning_lines.append(f"\n[{node_name}] - {timestamp_entry}")
            reasoning_lines.append("")
            # Indent the thought content
            for line in thought.split('\n'):
                reasoning_lines.append(f"  {line}")
            reasoning_lines.append("")

    reasoning_lines.append("=" * 80)
    reasoning_content = '\n'.join(reasoning_lines)

    # ========================================================================
    # 3. Save all three files
    # ========================================================================

    # Save the analysis report
    report_filename = "analysis_report.txt"
    try:
        with open(report_filename, 'w') as f:
            f.write(report_content)
        report_saved = True
    except Exception as e:
        print(f"Warning: Could not save report to file: {e}")
        report_saved = False

    # Save the reasoning log
    reasoning_filename = "agent_reasoning.txt"
    try:
        with open(reasoning_filename, 'w') as f:
            f.write(reasoning_content)
        reasoning_saved = True
    except Exception as e:
        print(f"Warning: Could not save reasoning to file: {e}")
        reasoning_saved = False

    # Save the code
    code_filename = "analysis_code.py"
    try:
        with open(code_filename, 'w') as f:
            f.write(state.get("code", ""))
        code_saved = True
    except Exception as e:
        print(f"Warning: Could not save code to file: {e}")
        code_saved = False

    # Log the final reasoning
    thought = f"""Generated final outputs and saved to files.

Files created:
1. {report_filename} - Analysis report with 3 sections (objective, details, conclusions)
   Status: {'Success' if report_saved else 'Failed'}

2. {reasoning_filename} - Complete agent reasoning process
   Status: {'Success' if reasoning_saved else 'Failed'}
   Contains: {len(reasoning_log)} thought entries across {len(iterations)} iteration(s)

3. {code_filename} - Python analysis code
   Status: {'Success' if code_saved else 'Failed'}"""

    reasoning_log_updated = add_thought(state, "generate_report", thought)

    return {
        **state,
        "reasoning_log": reasoning_log_updated
    }


def should_continue_decision(state: StatisticalAnalysisState) -> Literal["write_analysis_code", "generate_report"]:
    """Determine if we should continue iterating or move to report generation."""
    if state["should_continue"]:
        return "write_analysis_code"
    return "generate_report"


# Build the graph
def create_statistical_agent():
    """Create and compile the LangGraph statistical analysis agent."""

    workflow = StateGraph(StatisticalAnalysisState)

    # Add nodes
    workflow.add_node("understand_data", understand_data)
    workflow.add_node("plan_analysis", plan_analysis)
    workflow.add_node("write_analysis_code", write_analysis_code)
    workflow.add_node("execute_code", execute_code)
    workflow.add_node("validate_results", validate_results)
    workflow.add_node("generate_report", generate_report)

    # Add edges - linear flow from start through validation
    workflow.set_entry_point("understand_data")
    workflow.add_edge("understand_data", "plan_analysis")
    workflow.add_edge("plan_analysis", "write_analysis_code")
    workflow.add_edge("write_analysis_code", "execute_code")
    workflow.add_edge("execute_code", "validate_results")

    # Add conditional edge for iteration or completion
    workflow.add_conditional_edges(
        "validate_results",
        should_continue_decision,
        {
            "write_analysis_code": "write_analysis_code",  # Loop back to retry
            "generate_report": "generate_report"  # Move to report generation
        }
    )

    # Report generation is the final step
    workflow.add_edge("generate_report", END)

    return workflow.compile()


def run_statistical_analysis(task: str, data_file_path: str, output_dir: str = ".", max_iterations: int = 3):
    """Run the statistical analysis agent on a given task and data file.

    Args:
        task: The analysis task/question to answer
        data_file_path: Path to the data file to analyze
        output_dir: Directory to save plots and output files (default: current directory)
        max_iterations: Maximum number of code improvement iterations (default: 3)

    Returns:
        Final state dictionary containing all analysis results
    """

    # Create the agent
    agent = create_statistical_agent()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize state
    initial_state = {
        "task": task,
        "data_file_path": data_file_path,
        "output_dir": output_dir,
        "analysis_objective": "",
        "data_summary": "",
        "code": "",
        "execution_result": "",
        "execution_error": "",
        "analysis_details": "",
        "analysis_conclusions": "",
        "reasoning_log": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "should_continue": True,
        "is_valid": False
    }

    # Run the agent
    print(f"Starting Statistical Analysis Agent")
    print("=" * 80)
    print(f"Task: {task}")
    print(f"Data: {data_file_path}")
    print(f"Max iterations: {max_iterations}")
    print("=" * 80)
    print("\nAgent is working...\n")

    final_state = agent.invoke(initial_state)

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total iterations: {final_state['iteration']}")
    print(f"Analysis valid: {final_state['is_valid']}")
    print(f"\nOutputs generated:")
    print("  1. analysis_report.txt - Analysis report (3 sections)")
    print("  2. agent_reasoning.txt - Complete agent thought process")
    print("  3. analysis_code.py - Python code used for analysis")

    if final_state['execution_error']:
        print(f"\nWarning: Final execution had errors:")
        print(final_state['execution_error'][:200])

    print("\n" + "=" * 80)

    return final_state


if __name__ == "__main__":
    # Example usage

    # Make sure to set your OpenAI API key
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"

    # Example: Statistical analysis
    print("\n\nExample: Value at Risk (VaR) Analysis\n")

    # Note: You'll need to provide an actual data file path
    # For this example, we'll create a synthetic dataset first
    print("Creating sample data file for demonstration...\n")

    # Create a sample CSV file
    import pandas as pd
    import numpy as np

    # Generate sample returns data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    returns = np.random.normal(0.0005, 0.02, 1000)  # Mean return and volatility
    df = pd.DataFrame({
        'date': dates,
        'returns': returns
    })
    sample_data_path = 'sample_returns.csv'
    df.to_csv(sample_data_path, index=False)
    print(f"Sample data saved to: {sample_data_path}\n")

    # Run the statistical analysis
    run_statistical_analysis(
        task="Calculate the 99% Value at Risk (VaR) using both parametric and historical methods. Compare the results and create visualizations showing the return distribution, VaR threshold, and rolling VaR over time.",
        data_file_path=sample_data_path,
        max_iterations=3
    )

    print("\n\nAnalysis complete! Check the following files:")
    print("  1. analysis_report.txt - Final analysis report (3 sections)")
    print("  2. agent_reasoning.txt - Agent's complete thought process")
    print("  3. analysis_code.py - Python code used")
    print("  4. Any generated plots/figures")