import os
import weaviate
from weaviate.classes.query import MetadataQuery, Filter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import CrossEncoder

class WeaviateRetriever:
    """
    Core Retriever for fetching relevant contexts from Weaviate.
    Includes auto-detection of chunk strategy (hierarchical, sentence window, or recursive),
    relevance filtering, de-duplication, and Cross-Encoder reranking.
    """
    def __init__(self, config):
        self.config = config
        
        # 1. Initialize HuggingFaceEmbedding model for queries
        embed_model_name = config.get("embedding_model_name", "BAAI/bge-base-en-v1.5")
        print(f"[Retriever] Loading query embedding model: {embed_model_name}...")
        self.embed_model = HuggingFaceEmbedding(
            model_name=embed_model_name,
            query_instruction="Represent this sentence for searching relevant passages: "
        )
        
        # 2. Initialize CrossEncoder model for reranking
        rerank_model_name = config.get("rerank_model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        print(f"[Retriever] Loading Cross-Encoder reranker: {rerank_model_name}...")
        self.reranker = CrossEncoder(rerank_model_name)
        print("[Retriever] Models loaded successfully.")

    def retrieve_and_rerank(self, query_str, collection_name=None, limit=None, top_k=None, 
                            max_distance=None, alpha=None, min_score=None,
                            tenant_id=None, department_id=None):
        """
        Retrieves passages for a query from Weaviate using Hybrid search, applies strategy auto-detection,
        filters by relevance score, de-duplicates, and reranks using the Cross-Encoder.
        
        Args:
            query_str (str): The search query.
            collection_name (str): Weaviate collection name. Defaults to config value.
            limit (int): Number of objects to retrieve initially. Defaults to config value.
            top_k (int): Number of objects to return after reranking. Defaults to config value.
            max_distance (float): Kept for backward compatibility.
            alpha (float): Hybrid query alpha weight (0.0=keyword, 1.0=vector, 0.5=balanced).
            min_score (float): Minimum hybrid score threshold to keep a chunk.
            tenant_id (str): The client/tenant shard ID. Defaults to config value.
            department_id (str): Optional department metadata filter. Defaults to config value.
            
        Returns:
            list of dict: List of reranked passage objects with resolved context, scores, and metadata.
        """
        # Resolve configurations
        if collection_name is None:
            collection_name = self.config.get("weaviate_collection", "Documents")
        if limit is None:
            limit = self.config.get("retrieval_limit_before_rerank", 35)
        if top_k is None:
            top_k = self.config.get("rerank_top_k", 3)
        if alpha is None:
            alpha = self.config.get("hybrid_alpha", 0.5)
        if min_score is None:
            min_score = self.config.get("min_hybrid_score", 0.05)
        if tenant_id is None:
            tenant_id = self.config.get("tenant_id", "default_client")
        if department_id is None:
            department_id = self.config.get("department_id", "")
            
        # Connect to local Weaviate
        host = self.config.get("weaviate_host", "localhost")
        port = self.config.get("weaviate_port", 8080)
        grpc_port = self.config.get("weaviate_grpc_port", 50051)
        
        client = weaviate.connect_to_local(
            host=host,
            port=port,
            grpc_port=grpc_port
        )
        
        try:
            if not client.is_ready():
                raise ConnectionError("Weaviate is not reachable.")
                
            if not client.collections.exists(collection_name):
                print(f"[Retriever] Collection '{collection_name}' does not exist.")
                return []
                
            collection = client.collections.get(collection_name)
            
            # Check if tenant exists, if not return empty (no documents ingested yet for this tenant)
            try:
                # Get tenant collections
                tenant_collection = collection.with_tenant(tenant_id)
            except Exception as ex:
                print(f"[Retriever] Tenant shard '{tenant_id}' cannot be accessed: {ex}")
                return []
            
            # Step 1: Generate query embedding
            query_vector = self.embed_model.get_query_embedding(query_str)
            
            # Construct department filter if department_id is set
            filters = None
            if department_id:
                filters = Filter.by_property("department_id").equal(department_id)
            
            # Step 2: Query Weaviate using Hybrid search (vector + keyword)
            response = tenant_collection.query.hybrid(
                query=query_str,
                vector=query_vector,
                alpha=alpha,
                limit=limit,
                filters=filters,
                return_metadata=MetadataQuery(score=True, creation_time=True)
            )
            
            print(f"[Retriever] Tenant: '{tenant_id}', Dept: '{department_id or 'All'}'. Hybrid search retrieved {len(response.objects)} objects.")
            
            # Step 3: Strategy Auto-Detection, Context Resolution, Relevance Filtering, & De-duplication
            resolved_passages = []
            seen_texts = set()
            parent_cache = {} # Cache parent text to avoid redundant lookup requests
            
            for idx, obj in enumerate(response.objects):
                props = obj.properties
                score = obj.metadata.score or 0.0
                
                # Filter by relevance threshold (hybrid score)
                if score < min_score:
                    continue
                    
                parent_id = props.get("parent_id", "")
                window_text = props.get("window", "")
                chunk_text = props.get("text", "")
                
                resolved_text = ""
                resolved_strategy = "standard"
                
                # Dynamic Routing Heuristics
                if parent_id and parent_id.strip():
                    resolved_strategy = "hierarchical"
                    # Hierarchical Auto-Merging context resolution: fetch parent node
                    if parent_id in parent_cache:
                        resolved_text = parent_cache[parent_id]
                    else:
                        try:
                            parent_obj = collection.query.fetch_object_by_id(parent_id)
                            if parent_obj:
                                resolved_text = parent_obj.properties.get("text", "")
                                parent_cache[parent_id] = resolved_text
                            else:
                                # Fall back to child text if parent object is missing in DB
                                resolved_text = chunk_text
                        except Exception as e:
                            print(f"[Warning] Failed to fetch parent chunk {parent_id}: {e}")
                            resolved_text = chunk_text
                elif window_text and window_text.strip():
                    resolved_strategy = "sentence_window"
                    # Sentence Window context resolution: return expanded window text
                    resolved_text = window_text
                else:
                    resolved_strategy = "recursive"
                    # Recursive/Standard splitter: return chunk text directly
                    resolved_text = chunk_text
                    
                resolved_text = resolved_text.strip()
                if not resolved_text:
                    continue
                    
                # De-duplicate matching resolved texts
                if resolved_text in seen_texts:
                    continue
                seen_texts.add(resolved_text)
                
                resolved_passages.append({
                    "id": str(obj.uuid),
                    "text": resolved_text,
                    "original_text": chunk_text,
                    "source_file": props.get("source_file", ""),
                    "page_number": int(props.get("page_number", 0)),
                    "chunk_type": props.get("chunk_type", "text"),
                    "resolved_strategy": resolved_strategy,
                    "hybrid_score": score
                })

            print(f"[Retriever] Resolved & filtered down to {len(resolved_passages)} unique passages.")
            
            if not resolved_passages:
                return []
                
            # Step 4: Reranking with Cross-Encoder
            print(f"[Retriever] Scoring passages with Cross-Encoder...")
            # Form pairs of [query, passage]
            pairs = [[query_str, item["text"]] for item in resolved_passages]
            scores = self.reranker.predict(pairs)
            
            # Save cross encoder score and sort
            for item, score in zip(resolved_passages, scores):
                item["rerank_score"] = float(score)
                
            # Rank passages by cross-encoder score descending
            reranked_passages = sorted(resolved_passages, key=lambda x: x["rerank_score"], reverse=True)
            
            # Select Top K
            top_k_passages = reranked_passages[:top_k]
            print(f"[Retriever] Reranking complete. Selected top {len(top_k_passages)} passages.")
            return top_k_passages
            
        finally:
            client.close()
