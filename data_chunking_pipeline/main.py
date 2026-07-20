import os
import sys
# Add parent directory to sys.path to resolve data_chunking_pipeline imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
from data_chunking_pipeline.utils.helpers import load_config, validate_pdf_path
from data_chunking_pipeline.core.classifier import DocumentClassifier
from data_chunking_pipeline.core.extractor import IntelligentExtractor
from data_chunking_pipeline.core.chunker import (
    SentenceWindowStrategy,
    RecursiveSplitterStrategy,
    HierarchicalStrategy
)
from data_chunking_pipeline.core.uploader import WeaviateUploader

def parse_args():
    parser = argparse.ArgumentParser(description="Robust Multimodal Data Chunking Pipeline")
    parser.add_argument(
        "--pdf", 
        type=str, 
        required=True, 
        help="Path to the PDF file to chunk"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="data_chunking_pipeline/config.yaml", 
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--strategy", 
        type=str, 
        choices=["sentence_window", "recursive", "hierarchical"], 
        help="Override the chunking strategy configured in config.yaml"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Path to save the generated JSON chunks"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload generated chunks to Weaviate vector database"
    )
    parser.add_argument(
        "--overwrite_collection",
        action="store_true",
        help="Overwrite the Weaviate collection if it exists"
    )
    parser.add_argument(
        "--tenant", 
        type=str, 
        help="The client/tenant shard ID to upload for"
    )
    parser.add_argument(
        "--department", 
        type=str, 
        help="Optional sub-tenant department filter metadata"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load config and validate inputs
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config file '{args.config}': {e}")
        sys.exit(1)
        
    pdf_path = args.pdf
    try:
        validate_pdf_path(pdf_path)
    except Exception as e:
        print(f"Invalid PDF path: {e}")
        sys.exit(1)

    strategy_name = args.strategy or config.get("chunk_strategy", "sentence_window")
    print(f"\n=======================================================")
    print(f"[INIT] Initializing Chunking Pipeline")
    print(f"  - Input PDF: {pdf_path}")
    print(f"  - Strategy: {strategy_name}")
    print(f"=======================================================\n")
    
    # 2. Phase 1: Classification
    classifier = DocumentClassifier(
        ocr_threshold_chars=config.get("ocr_threshold_chars_per_page", 100),
        table_density_threshold=config.get("table_density_threshold", 0.5)
    )
    
    doc_class = classifier.classify(pdf_path)
    print(f"Routing logic completed. Classification: {doc_class}\n")
    
    # 3. Phase 2: Extraction
    extractor = IntelligentExtractor(
        table_preceding_lines=config.get("table_preceding_context_lines", 5),
        table_succeeding_lines=config.get("table_succeeding_context_lines", 5)
    )
    extracted_pages = extractor.extract_document(pdf_path, classification=doc_class)
    print(f"Extraction completed. Parsed {len(extracted_pages)} pages.")
    
    # 4. Phase 3: Adaptive Chunking
    strategy_map = {
        "sentence_window": SentenceWindowStrategy,
        "recursive": RecursiveSplitterStrategy,
        "hierarchical": HierarchicalStrategy
    }
    
    chunker_class = strategy_map.get(strategy_name)
    if not chunker_class:
        print(f"Unsupported strategy: {strategy_name}")
        sys.exit(1)
        
    chunker = chunker_class(**config)
    source_filename = os.path.basename(pdf_path)
    
    nodes = chunker.chunk(extracted_pages, source_file=source_filename)
    
    # Inject department metadata if provided
    dept_id = args.department or config.get("department_id", "")
    if dept_id:
        for node in nodes:
            node.metadata["department_id"] = dept_id
            
    print(f"Chunking completed. Generated {len(nodes)} chunks/nodes.\n")
    
    # Print summary statistics
    text_chunks_count = sum(1 for n in nodes if n.metadata.get("chunk_type") == "text")
    table_chunks_count = sum(1 for n in nodes if n.metadata.get("chunk_type") == "table")
    print(f"Summary of Generated Nodes:")
    print(f"  - Total Nodes: {len(nodes)}")
    print(f"  - Text Nodes: {text_chunks_count}")
    print(f"  - Table Nodes: {table_chunks_count}")
    
    # Render preview of first few nodes
    print(f"\nPreview of first 3 nodes:")
    for idx, node in enumerate(nodes[:3]):
        print(f"--- Node {idx+1} ({node.metadata.get('chunk_type', 'unknown')}) | Page: {node.metadata.get('page_number')} ---")
        preview = node.text[:250].replace('\n', ' ')
        print(f"Text Preview: {preview}...")
        print(f"Metadata: {node.metadata}")
        if hasattr(node, "relationships") and node.relationships:
            # show simple view of relationships
            rels = {str(k): v.node_id if hasattr(v, 'node_id') else str(v) for k, v in node.relationships.items()}
            print(f"Relationships: {rels}")
        print()
    
    # 5. Save output if requested
    if args.output:
        serializable_nodes = []
        for node in nodes:
            node_dict = {
                "id": node.node_id,
                "text": node.text,
                "metadata": node.metadata,
                "relationships": {}
            }
            if hasattr(node, "relationships") and node.relationships:
                for rel_type, rel_info in node.relationships.items():
                    # Check if list of nodes or single node
                    if isinstance(rel_info, list):
                        node_dict["relationships"][str(rel_type)] = [r.node_id for r in rel_info]
                    elif hasattr(rel_info, "node_id"):
                        node_dict["relationships"][str(rel_type)] = rel_info.node_id
                    else:
                        node_dict["relationships"][str(rel_type)] = str(rel_info)
            serializable_nodes.append(node_dict)
            
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(serializable_nodes, f, indent=2, ensure_ascii=False)
        print(f"[SAVE] Successfully saved {len(nodes)} chunks to {args.output}")

    # 6. Upload to Weaviate if requested
    if args.upload or config.get("upload_to_weaviate", False):
        try:
            overwrite = args.overwrite_collection or config.get("overwrite_collection", False)
            uploader = WeaviateUploader(config)
            uploader.upload_chunks(
                nodes=nodes,
                overwrite=overwrite,
                tenant_id=args.tenant,
                department_id=args.department
            )
        except Exception as e:
            print(f"[Uploader] Error during Weaviate upload: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
