from abc import ABC, abstractmethod
from llama_index.core import Document
from llama_index.core.schema import TextNode, NodeRelationship
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser, SentenceSplitter

class BaseChunkingStrategy(ABC):
    """
    Abstract Base Class for Chunking Strategies.
    """
    @abstractmethod
    def chunk(self, extracted_pages, source_file, **kwargs):
        """
        Chunks the page elements into a list of TextNode objects.
        
        Args:
            extracted_pages (list): List of pages as returned by IntelligentExtractor.
            source_file (str): The filename/path of the PDF.
            
        Returns:
            list: List of TextNode objects.
        """
        pass


class SentenceWindowStrategy(BaseChunkingStrategy):
    """
    Sentence Window strategy. Splits text into individual sentences but keeps
    surrounding sentence context as window metadata. Tables are kept atomic.
    """
    def __init__(self, window_size=3, **kwargs):
        self.window_size = window_size
        self.parser = SentenceWindowNodeParser.from_defaults(
            window_size=self.window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        )

    def chunk(self, extracted_pages, source_file, **kwargs):
        chunked_nodes = []
        
        for page_data in extracted_pages:
            page_num = page_data['page_number']
            
            for elem in page_data['elements']:
                if elem['type'] == 'text':
                    # Use llama-index SentenceWindowNodeParser on plain text
                    doc = Document(text=elem['retrieved_text'])
                    nodes = self.parser.get_nodes_from_documents([doc])
                    
                    for node in nodes:
                        node.metadata.update({
                            'source_file': source_file,
                            'page_number': page_num,
                            'chunk_type': 'text',
                            'parent_id': None
                        })
                        chunked_nodes.append(node)
                        
                elif elem['type'] in ['table', 'code']:
                    # Tables and code blocks are atomic, but we wrap/anchor them
                    table_md = elem['markdown']
                    table_text = elem['retrieved_text']
                    
                    node = TextNode(text=table_md)
                    node.metadata = {
                        'source_file': source_file,
                        'page_number': page_num,
                        'chunk_type': elem['type'],
                        'parent_id': None,
                        # Match sentence-window schema so downstream index can read it
                        'window': table_text,
                        'original_text': table_md
                    }
                    chunked_nodes.append(node)
                    
        return chunked_nodes


class RecursiveSplitterStrategy(BaseChunkingStrategy):
    """
    Recursive Character Splitter strategy. Splits text recursively using 
    separators while keeping paragraphs intact. Tables are kept atomic.
    """
    def __init__(self, chunk_size=512, chunk_overlap=64, **kwargs):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def chunk(self, extracted_pages, source_file, **kwargs):
        chunked_nodes = []
        
        for page_data in extracted_pages:
            page_num = page_data['page_number']
            
            for elem in page_data['elements']:
                if elem['type'] == 'text':
                    doc = Document(text=elem['retrieved_text'])
                    nodes = self.parser.get_nodes_from_documents([doc])
                    
                    for node in nodes:
                        node.metadata.update({
                            'source_file': source_file,
                            'page_number': page_num,
                            'chunk_type': 'text',
                            'parent_id': None
                        })
                        chunked_nodes.append(node)
                        
                elif elem['type'] in ['table', 'code']:
                    table_text = elem['retrieved_text']
                    
                    node = TextNode(text=table_text)
                    node.metadata = {
                        'source_file': source_file,
                        'page_number': page_num,
                        'chunk_type': elem['type'],
                        'parent_id': None
                    }
                    chunked_nodes.append(node)
                    
        return chunked_nodes


class HierarchicalStrategy(BaseChunkingStrategy):
    """
    Hierarchical Parent-Child splitter. Generates multi-level text hierarchy.
    Tables are kept atomic.
    """
    def __init__(self, hierarchical_chunk_sizes=None, **kwargs):
        if hierarchical_chunk_sizes is None:
            hierarchical_chunk_sizes = [1024, 512, 128]
        self.chunk_sizes = hierarchical_chunk_sizes
        self.parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=self.chunk_sizes
        )

    def chunk(self, extracted_pages, source_file, **kwargs):
        chunked_nodes = []
        
        for page_data in extracted_pages:
            page_num = page_data['page_number']
            
            for elem in page_data['elements']:
                if elem['type'] == 'text':
                    doc = Document(text=elem['retrieved_text'])
                    nodes = self.parser.get_nodes_from_documents([doc])
                    
                    for node in nodes:
                        # Extract parent_id from relationships
                        parent_rel = node.relationships.get(NodeRelationship.PARENT)
                        parent_id = parent_rel.node_id if parent_rel else None
                        
                        node.metadata.update({
                            'source_file': source_file,
                            'page_number': page_num,
                            'chunk_type': 'text',
                            'parent_id': parent_id
                        })
                        chunked_nodes.append(node)
                        
                elif elem['type'] in ['table', 'code']:
                    table_text = elem['retrieved_text']
                    
                    node = TextNode(text=table_text)
                    node.metadata = {
                        'source_file': source_file,
                        'page_number': page_num,
                        'chunk_type': elem['type'],
                        'parent_id': None
                    }
                    chunked_nodes.append(node)
                    
        return chunked_nodes
