import os
import sys
import argparse
# Add parent directory to sys.path to resolve data_chunking_pipeline imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_chunking_pipeline.utils.helpers import load_config
from data_chunking_pipeline.core.retriever import WeaviateRetriever
from data_chunking_pipeline.core.generator import LLMGenerator

def parse_args():
    parser = argparse.ArgumentParser(description="Query Retrieval and Reranker Pipeline")
    parser.add_argument(
        "--query", 
        type=str, 
        required=True, 
        help="Semantic search query string"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="data_chunking_pipeline/config.yaml", 
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--collection", 
        type=str, 
        help="Override the Weaviate collection name"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        help="Override the initial vector search limit"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        help="Override the top K rerank limit"
    )
    parser.add_argument(
        "--max_distance", 
        type=float, 
        help="Override the maximum cosine distance threshold"
    )
    parser.add_argument(
        "--llm", 
        type=str, 
        choices=["openai", "local", "none"],
        help="Override the LLM provider (openai, local, or none)"
    )
    parser.add_argument(
        "--openai-key", 
        type=str, 
        help="Override/provide OpenAI API Key"
    )
    parser.add_argument(
        "--local-model", 
        type=str, 
        help="Override the local HuggingFace model path"
    )
    parser.add_argument(
        "--alpha", 
        type=float, 
        help="Override the hybrid search alpha value (0.0=keyword, 1.0=vector)"
    )
    parser.add_argument(
        "--tenant", 
        type=str, 
        help="The client/tenant shard ID to query"
    )
    parser.add_argument(
        "--department", 
        type=str, 
        help="Optional sub-tenant department filter metadata"
    )
    return parser.parse_args()
 
def main():
    args = parse_args()
    
    # 1. Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config file '{args.config}': {e}")
        sys.exit(1)
        
    print(f"\n=======================================================")
    print(f"[QUERY] Initializing Query Pipeline")
    print(f"  - Query: '{args.query}'")
    print(f"=======================================================\n")
    
    # 2. Run retrieval & reranking
    try:
        retriever = WeaviateRetriever(config)
        results = retriever.retrieve_and_rerank(
            query_str=args.query,
            collection_name=args.collection,
            limit=args.limit,
            top_k=args.top_k,
            max_distance=args.max_distance,
            alpha=args.alpha,
            tenant_id=args.tenant,
            department_id=args.department
        )
    except Exception as e:
        print(f"[Error] Retrieval failed: {e}")
        sys.exit(1)
        
    if not results:
        print("[Retriever] No relevant passages found matching the criteria.")
        sys.exit(0)
        
    # 3. Print ranked results details
    print(f"\n=======================================================")
    print(f"Ranked Contexts (Top {len(results)} after Cross-Encoder):")
    print(f"=======================================================")
    
    for idx, item in enumerate(results):
        print(f"\n[Rank {idx+1}] Score: {item['rerank_score']:.4f} (Weaviate Score: {item['hybrid_score']:.4f})")
        print(f"  - Source: {item['source_file']} | Page: {item['page_number']} | Type: {item['chunk_type']}")
        print(f"  - Auto-Detected Strategy: {item['resolved_strategy']}")
        print(f"  - Passage Context:")
        # Indent lines of text
        indented_text = "\n".join("      " + line for line in item["text"].split("\n"))
        print(indented_text)
        print("-" * 50)
        
    # 4. Print consolidated block for LLM prompt context insertion
    print(f"\n=======================================================")
    print(f"Formatted LLM Prompt Context:")
    print(f"=======================================================\n")
    
    prompt_context = "<context>\n"
    for idx, item in enumerate(results):
        prompt_context += f"Document [{idx+1}]: Source: {item['source_file']} (Page {item['page_number']}, Type: {item['chunk_type']})\n"
        prompt_context += f"{item['text']}\n"
        prompt_context += f"====================\n"
    prompt_context += "</context>"
    
    print(prompt_context)
    print("\n=======================================================\n")

    # 5. LLM Answer Generation
    provider = args.llm or config.get("llm_provider", "none")
    if provider.lower() != "none":
        config["llm_provider"] = provider # Ensure generator gets the overridden provider
        print(f"=======================================================")
        print(f"[LLM] Generating answer using provider: {provider.upper()}")
        print(f"=======================================================\n")
        try:
            generator = LLMGenerator(config, openai_api_key=args.openai_key, local_model_name=args.local_model)
            answer = generator.generate_answer(args.query, prompt_context)
            print(f"Answer:\n{answer}\n")
        except Exception as e:
            print(f"[Error] Generation failed: {e}\n")
        print(f"=======================================================\n")

if __name__ == "__main__":
    main()
