import pandas as pd
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

from langfuse.decorators import observe, langfuse_context

@observe(as_type="generation", name="table_agent")
def table_agent(query: str, df: pd.DataFrame) -> dict:
    schema = "\n".join([f"- {col}: type {df[col].dtype}" for col in df.columns])
    sample_rows = df.head(3).to_string()
    
    prompt = f"""You are a data analysis bot. You have a pandas DataFrame named 'df'.
DataFrame columns and types:
{schema}

Sample rows:
{sample_rows}

Write a single Python/pandas expression that answers this question: "{query}"
Do not write a full program or a multi-line script. Write ONLY a single expression that evaluates to the answer.
Do not include any explanation, imports, markdown blocks, or variables other than 'df' and 'pd'.

Examples:
- Question: "What is the average cost of all active devices?" -> df[df['status'] == 'Active']['cost'].mean()
- Question: "Who is ASSET_001 assigned to?" -> df[df['asset_id'] == 'ASSET_001']['assigned_to'].iloc[0]

Answer:"""

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    
    content = resp.choices[0].message.content
    code = content.strip()
    
    langfuse_context.update_current_observation(
        input=prompt,
        output=content,
        model="llama-3.1-8b-instant",
        usage={
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens
        }
    )
    
    # Clean up code blocks if the LLM outputted them
    if code.startswith("```"):
        lines = code.split("\n")
        # Remove starting ```python or ```
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove ending ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines).strip()
        
    # Strip any prefix like "python"
    if code.lower().startswith("python"):
        code = code[6:].strip()
        
    try:
        # Evaluate the pandas expression in a context where 'df' and 'pd' are defined
        result = eval(code, {"df": df, "pd": pd})
        return {
            "answer": str(result),
            "code": code
        }
    except Exception as e:
        return {
            "answer": f"Could not compute: {e}",
            "code": code
        }

if __name__ == "__main__":
    csv_path = "data/sample_table.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        queries = [
            "What is the total cost of all active devices in engineering?",
            "How many devices are in repair?",
            "Who is ASSET_010 assigned to?"
        ]
        for q in queries:
            res = table_agent(q, df)
            print(f"Query: {q}\nPandas Code: {res['code']}\nAnswer: {res['answer']}\n")
    else:
        print(f"CSV file not found at {csv_path}")
