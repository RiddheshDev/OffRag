import os
import weaviate
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter
from weaviate.classes.tenants import Tenant
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class WeaviateUploader:
    """
    Component to generate dense embeddings for chunks and upload them
    to a Weaviate database with explicit schema definitions.
    """
    def __init__(self, config):
        """
        Args:
            config (dict): Configuration dictionary containing model name and Weaviate parameters.
        """
        self.config = config
        
        # 1. Initialize HuggingFaceEmbedding model
        model_name = config.get("embedding_model_name", "BAAI/bge-base-en-v1.5")
        print(f"[Uploader] Initializing embedding model: {model_name}...")
        self.embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            query_instruction="Represent this sentence for searching relevant passages: "
        )
        print("[Uploader] Embedding model initialized successfully.")

    def upload_chunks(self, nodes, collection_name=None, overwrite=False, tenant_id=None, department_id=None):
        """
        Generates embeddings for each node and uploads them to Weaviate.
        
        Args:
            nodes (list): List of TextNode objects.
            collection_name (str): The name of the Weaviate collection. Defaults to config value.
            overwrite (bool): If True, deletes the collection if it exists before creating it.
            tenant_id (str): The client/tenant shard ID. Defaults to config value.
            department_id (str): Optional department metadata filter. Defaults to config value.
        """
        if not nodes:
            print("[Uploader] No nodes to upload.")
            return []

        # Resolve config overrides
        if tenant_id is None:
            tenant_id = self.config.get("tenant_id", "default_client")
        if department_id is None:
            department_id = self.config.get("department_id", "")

        # 2. Establish connection to local Weaviate instance
        host = self.config.get("weaviate_host", "localhost")
        port = self.config.get("weaviate_port", 8080)
        grpc_port = self.config.get("weaviate_grpc_port", 50051)
        
        print(f"[Uploader] Connecting to Weaviate at {host}:{port} (gRPC: {grpc_port})...")
        client = weaviate.connect_to_local(
            host=host,
            port=port,
            grpc_port=grpc_port
        )
        
        try:
            if not client.is_ready():
                raise ConnectionError("Weaviate instance is not ready or accessible.")
                
            if collection_name is None:
                collection_name = self.config.get("weaviate_collection", "Documents")
                
            print(f"[Uploader] Target collection: '{collection_name}' (tenant_id='{tenant_id}', overwrite={overwrite})")
            
            # Delete if overwrite is set and collection exists
            if overwrite and client.collections.exists(collection_name):
                print(f"[Uploader] Deleting existing collection '{collection_name}'...")
                client.collections.delete(collection_name)
                
            # Create collection with explicit properties & multi-tenancy if it doesn't exist
            if not client.collections.exists(collection_name):
                print(f"[Uploader] Creating collection '{collection_name}' with Multi-Tenancy enabled...")
                properties = [
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="source_file", data_type=DataType.TEXT),
                    Property(name="page_number", data_type=DataType.INT),
                    Property(name="chunk_type", data_type=DataType.TEXT),
                    Property(name="parent_id", data_type=DataType.TEXT),
                    Property(name="window", data_type=DataType.TEXT),
                    Property(name="department_id", data_type=DataType.TEXT),
                ]
                collection = client.collections.create(
                    name=collection_name,
                    multi_tenancy_config=Configure.multi_tenancy(enabled=True),
                    vectorizer_config=Configure.Vectorizer.none(),
                    properties=properties
                )
            else:
                collection = client.collections.get(collection_name)

            # Ensure the specific tenant exists on the collection
            try:
                collection.tenants.create([Tenant(name=tenant_id)])
                print(f"[Uploader] Shard for tenant '{tenant_id}' initialized.")
            except Exception:
                # Tenant shard already exists, ignore
                pass

            # Bind client shard for this tenant session
            tenant_collection = collection.with_tenant(tenant_id)
                
            # Incremental overwrite: Delete old chunks of the active files on this tenant's shard
            if not overwrite:
                unique_files = set(node.metadata.get("source_file") for node in nodes if node.metadata.get("source_file"))
                for filename in unique_files:
                    print(f"[Uploader] [Tenant: {tenant_id}] Cleaning up old chunks for file: {filename}...")
                    tenant_collection.data.delete_many(
                        where=Filter.by_property("source_file").equal(filename)
                    )
                
            # 3. Generate embeddings and construct DataObjects
            print(f"[Uploader] Generating embeddings for {len(nodes)} chunks...")
            data_objects = []
            
            for idx, node in enumerate(nodes):
                # For both text and table nodes, we embed the node's main text content.
                # In sentence window nodes, the main text is the target sentence, and 
                # the larger window is saved in properties.
                text_to_embed = node.text
                
                # Fetch text embedding
                embedding = self.embed_model.get_text_embedding(text_to_embed)
                
                # Construct Weaviate properties
                props = {
                    "text": node.text,
                    "source_file": node.metadata.get("source_file", ""),
                    "page_number": int(node.metadata.get("page_number", 0)),
                    "chunk_type": node.metadata.get("chunk_type", "text"),
                    "parent_id": node.metadata.get("parent_id") or "",
                    "window": node.metadata.get("window") or "",
                    "department_id": node.metadata.get("department_id") or department_id
                }
                
                # Prepare Weaviate DataObject
                data_objects.append(DataObject(
                    properties=props,
                    vector=embedding,
                    uuid=node.node_id
                ))
                
                if (idx + 1) % 50 == 0 or idx == len(nodes) - 1:
                    print(f"  - Embedded {idx + 1}/{len(nodes)} chunks")
                    
            # 4. Perform batch upload
            print(f"[Uploader] Uploading {len(data_objects)} objects to Weaviate in batch...")
            res = tenant_collection.data.insert_many(data_objects)
            
            if res.has_errors:
                print("[Uploader] Error: Batch upload had errors:")
                for err in res.errors[:5]:
                    print(f"  - UUID {err.uuid_}: {err.message}")
                if len(res.errors) > 5:
                    print(f"  - and {len(res.errors) - 5} more errors.")
                raise RuntimeError("Weaviate batch upload encountered errors.")
            else:
                print(f"[Uploader] Ingestion completed. Ingested {len(data_objects)} objects.")
                
            return [obj.uuid for obj in data_objects]
            
        finally:
            client.close()
            print("[Uploader] Weaviate connection closed.")
