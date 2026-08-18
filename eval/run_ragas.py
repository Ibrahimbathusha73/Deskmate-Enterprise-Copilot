import json
import time
import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from orchestrator.graph import deskmate_graph

# Load environment variables
load_dotenv()

import argparse

def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run evaluation on full golden set")
    # Parse args while ignoring unrecognized args from other frameworks/runners
    args, unknown = parser.parse_known_args()
    
    print("Loading golden set...")
    with open("eval/golden_set.json") as f:
        golden = json.load(f)
        
    if not args.full:
        print("Running in quick/CI mode (3 representative queries). Pass --full for all 30 queries.")
        # Select indices: 0 (docs), 15 (table), and 23 (ticket)
        golden = [golden[0], golden[15], golden[23]]
    
    print(f"Loaded {len(golden)} evaluation items.")
    records = []
    
    # Process queries through the multi-agent graph
    for idx, item in enumerate(golden):
        print(f"[{idx+1}/{len(golden)}] Processing: '{item['question']}' (Agent: {item['agent']})")
        
        # Throttling to prevent Groq API rate limiting
        time.sleep(1.0)
        
        try:
            result = deskmate_graph.invoke({"query": item["question"]})
            
            # Retrieve answer and truncate
            answer = result.get("answer", "")
            if len(answer) > 1000:
                answer = answer[:1000]
            
            # Retrieve contexts and truncate
            chunks = result.get("retrieved_chunks") or []
            contexts = [c["text"][:1000] for c in chunks] if chunks else [""]
            
            # Retrieve expected answer and truncate
            gt = item["expected_answer"]
            if len(gt) > 1000:
                gt = gt[:1000]
            
            records.append({
                "question": item["question"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": gt,
            })
        except Exception as e:
            print(f"Error invoking graph for question '{item['question']}': {e}")
            # Append fallback so the dataset remains aligned
            records.append({
                "question": item["question"],
                "answer": f"Error: {e}",
                "contexts": [""],
                "ground_truth": item["expected_answer"][:1000] if item["expected_answer"] else "",
            })
            
    print("\nGraph execution complete. Converting to Hugging Face Dataset...")
    ds = Dataset.from_list(records)
    
    print("Initializing Groq LLM Judge and local HuggingFace embeddings for Ragas...")
    # Use Llama 3.1 8B on Groq for the judge
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    # Bypass n parameter to support Groq's API constraints and supply custom is_finished_parser
    evaluator_llm = LangchainLLMWrapper(llm, bypass_n=True, is_finished_parser=lambda x: True)
    
    # Use standard LangChain HuggingFaceEmbeddings wrapped for Ragas compatibility
    langchain_emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    evaluator_embeddings = LangchainEmbeddingsWrapper(langchain_emb)
    
    from ragas.run_config import RunConfig
    
    print("Running Ragas evaluation...")
    run_cfg = RunConfig(max_workers=1)
    scores = evaluate(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_cfg
    )
    
    import numpy as np
    
    # Calculate custom aggregated scores to avoid non-docs questions (which have no contexts) dragging down RAG faithfulness/precision/recall metrics.
    faithfulness_scores = []
    context_precision_scores = []
    context_recall_scores = []
    answer_relevancy_scores = []
    
    # scores.scores is a list of dictionaries, one per row
    row_scores_list = scores.scores
    for i, row in enumerate(records):
        has_context = any(c != "" and c is not None for c in row["contexts"])
        row_scores = row_scores_list[i] if i < len(row_scores_list) else {}
        
        ar = row_scores.get("answer_relevancy")
        if ar is not None and not np.isnan(ar):
            answer_relevancy_scores.append(ar)
            
        if has_context:
            f_score = row_scores.get("faithfulness")
            if f_score is not None and not np.isnan(f_score):
                faithfulness_scores.append(f_score)
                
            cp = row_scores.get("context_precision")
            if cp is not None and not np.isnan(cp):
                context_precision_scores.append(cp)
                
            cr = row_scores.get("context_recall")
            if cr is not None and not np.isnan(cr):
                context_recall_scores.append(cr)
                
    custom_summary = {
        "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0,
        "answer_relevancy": sum(answer_relevancy_scores) / len(answer_relevancy_scores) if answer_relevancy_scores else 0.0,
        "context_precision": sum(context_precision_scores) / len(context_precision_scores) if context_precision_scores else 0.0,
        "context_recall": sum(context_recall_scores) / len(context_recall_scores) if context_recall_scores else 0.0,
    }
    
    print("\nEvaluation completed. Custom Aggregated Scores (context-based metrics filtered to docs questions):")
    print(custom_summary)
    
    # Save scores and details for analysis
    output_data = {
        "summary": custom_summary,
        "details": records
    }
    
    os.makedirs("eval", exist_ok=True)
    with open("eval/last_run_scores.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print("Results saved to eval/last_run_scores.json")

if __name__ == "__main__":
    run_evaluation()
