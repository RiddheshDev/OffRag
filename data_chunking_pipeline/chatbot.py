import os
import sys
import json
import re
import argparse

# Add parent directory to sys.path to resolve imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_chunking_pipeline.utils.helpers import load_config
from data_chunking_pipeline.core.retriever import WeaviateRetriever
from data_chunking_pipeline.core.generator import LLMGenerator
from data_chunking_pipeline.core.mcp_client import MCPClient

class ConversationalChatbot:
    """
    Production-grade Conversational Agent utilizing:
    1. Dynamic Intent Routing (Approach 2)
    2. Custom Weaviate Retriever (Hybrid Search + Reranking)
    3. MCP Search Fallback (Agentic RAG)
    4. Chat History Memory Management
    """
    def __init__(self, config_path, openai_key=None, local_model=None, llm_provider=None, tenant_id=None, department_id=None):
        self.config = load_config(config_path)
        if llm_provider:
            self.config["llm_provider"] = llm_provider
        if tenant_id:
            self.config["tenant_id"] = tenant_id
        if department_id:
            self.config["department_id"] = department_id
            
        # 1. Initialize Components
        self.retriever = WeaviateRetriever(self.config)
        self.generator = LLMGenerator(self.config, openai_api_key=openai_key, local_model_name=local_model)
        self.mcp_client = MCPClient(self.config)
        
        # 2. Initialize Conversation Memory
        self.chat_history = []
        self.history_limit = self.config.get("chat_history_limit", 5)
        self.mcp_threshold = self.config.get("mcp_fallback_threshold", 0.1)

    def add_message(self, role, content):
        self.chat_history.append({"role": role, "content": content})
        # Slide history window
        if len(self.chat_history) > self.history_limit * 2:
            self.chat_history = self.chat_history[-self.history_limit * 2:]

    def _get_history_str(self):
        history_lines = []
        for msg in self.chat_history:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{role_label}: {msg['content']}")
        return "\n".join(history_lines)

    def route_intent(self, user_message):
        """
        Classifies user query intent (Approach 2) and condenses conversational follow-ups.
        
        Returns:
            tuple: (needs_search: bool, standalone_query: str)
        """
        if not self.chat_history:
            # First message: always search if it contains technical queries,
            # but do a quick check to skip if it's a simple greeting.
            greetings = ["hello", "hi", "hey", "hola", "greetings", "good morning", "good afternoon"]
            clean_msg = user_message.strip().lower().replace(".", "").replace("!", "")
            if clean_msg in greetings:
                return False, ""
            return True, user_message
            
        history_str = self._get_history_str()
        
        condense_prompt = (
            "You are a conversation intent router. Analyze the conversation history and the new user message.\n"
            "Determine if answering the new user message requires searching the reference document database for factual details.\n"
            "Respond ONLY with a JSON object in this exact format:\n"
            "{\n"
            '  "needs_search": true or false,\n'
            '  "standalone_query": "A rewritten search query, or empty string if needs_search is false."\n'
            "}\n\n"
            "Guidelines:\n"
            "- Set needs_search to false for greetings, thank yous, or requests to format/explain/summarize the previous reply.\n"
            "- Set needs_search to true if the user asks a new question about URLs, APIs, codes, specifications, or facts.\n\n"
            "Examples:\n"
            "1. History:\n"
            "User: Hello!\n"
            "Assistant: Hi! How can I help?\n"
            "Message: What is the URL for Language Detection?\n"
            "Response:\n"
            "{\n"
            '  "needs_search": true,\n'
            '  "standalone_query": "What is the URL for Language Detection?"\n'
            "}\n\n"
            "2. History:\n"
            "User: What is the URL?\n"
            "Assistant: The URL is http://api.datumbox.com.\n"
            "Message: Can you format that simply?\n"
            "Response:\n"
            "{\n"
            '  "needs_search": false,\n'
            '  "standalone_query": ""\n'
            "}\n\n"
            f"Active History:\n{history_str}\n\n"
            f"Active Message: {user_message}\n\n"
            "Response:"
        )
        
        try:
            # Query LLM to condense and route using raw generation (bypassing RAG constraints)
            raw_response = self.generator.generate_raw(
                system_prompt="You are a JSON assistant. Output only raw JSON.",
                user_prompt=condense_prompt
            )
            
            # Simple JSON parsing regex safety
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                needs_search = bool(data.get("needs_search", True))
                standalone_query = data.get("standalone_query", "").strip()
                
                # Self-healing heuristic for tiny models:
                # If standalone_query is generated and has content, but needs_search is false:
                # check if user message is just simple chitchat. If not, override needs_search to true.
                if not needs_search and standalone_query:
                    chitchat_patterns = [r"^hello\b", r"^hi\b", r"^thanks\b", r"^thank\b", r"^ok\b", r"^cool\b", r"^bye\b"]
                    clean_msg = user_message.strip().lower()
                    is_chitchat = any(re.search(pat, clean_msg) for pat in chitchat_patterns)
                    if not is_chitchat and len(standalone_query.split()) > 2:
                        needs_search = True
                        
                if not standalone_query:
                    standalone_query = user_message
                    
                return needs_search, standalone_query
                
        except Exception as e:
            print(f"[Router Warning] Failed to route intent using LLM: {e}")
            
        # Safe fallback: Search Weaviate using raw query
        return True, user_message

    def get_response(self, user_message):
        """
        Processes conversational turn: resolves intent, runs retrieval + MCP fallback,
        adds context, and generates answer.
        """
        # Step 1: Resolve Intent & Standalone Query (Approach 2)
        needs_search, standalone_query = self.route_intent(user_message)
        
        context_str = ""
        top_score = 0.0
        
        if needs_search:
            print(f"\n[Router] Search required. Condensed standalone query: '{standalone_query}'")
            # Step 2: Query custom Weaviate retriever (hybrid search, parent merging, reranking)
            results = self.retriever.retrieve_and_rerank(
                query_str=standalone_query,
                tenant_id=self.config.get("tenant_id"),
                department_id=self.config.get("department_id")
            )
            
            if results:
                top_score = results[0].get("rerank_score", -99.0)
                
            # Step 3: Check relevance threshold for Agentic MCP Fallback
            if not results or top_score < self.mcp_threshold:
                print(f"[Router] Retrieval confidence below threshold ({top_score:.4f} < {self.mcp_threshold}). Triggering MCP Search fallback...")
                search_results = self.mcp_client.run_search(standalone_query)
                
                if search_results:
                    context_str = "<context>\n"
                    for idx, web_res in enumerate(search_results):
                        context_str += f"Document [Web {idx+1}]: Source: {web_res['url']} (Search Fallback)\n"
                        context_str += f"{web_res['snippet']}\n"
                        context_str += "====================\n"
                    context_str += "</context>"
                else:
                    context_str = "No reference database chunks or web results were found."
            else:
                # Format Weaviate results with bracket indexes
                print(f"[Router] DB search matches found! Top Rerank Score: {top_score:.4f}")
                context_str = "<context>\n"
                for idx, item in enumerate(results):
                    context_str += f"Document [{idx+1}]: Source: {item['source_file']} (Page {item['page_number']}, Type: {item['chunk_type']})\n"
                    context_str += f"{item['text']}\n"
                    context_str += "====================\n"
                context_str += "</context>"
        else:
            print(f"\n[Router] Search skipped. Chat history will be used.")
            context_str = "No database search was required for this conversational turn. Rely on previous discussion or chat history."

        # Step 4: Inject chat history into system prompt context
        history_str = self._get_history_str()
        prompt_with_history = f"Chat History:\n{history_str}\n\nCurrent Context:\n{context_str}"
        
        # Step 5: Generate answer
        answer = self.generator.generate_answer(user_message, prompt_with_history)
        
        # Step 6: Update conversation state
        self.add_message("user", user_message)
        self.add_message("assistant", answer)
        
        return answer

def parse_args():
    parser = argparse.ArgumentParser(description="Conversational RAG Chatbot")
    parser.add_argument(
        "--config", 
        type=str, 
        default="data_chunking_pipeline/config.yaml", 
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--llm", 
        type=str, 
        choices=["openai", "local", "none"],
        help="Override the LLM provider"
    )
    parser.add_argument(
        "--openai-key", 
        type=str, 
        help="Override/provide OpenAI API Key"
    )
    parser.add_argument(
        "--local-model", 
        type=str, 
        help="Override the local model path"
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
    
    print("\n=======================================================")
    print("Starting Conversational Chatbot Engine")
    print("=======================================================")
    
    chatbot = ConversationalChatbot(
        config_path=args.config,
        openai_key=args.openai_key,
        local_model=args.local_model,
        llm_provider=args.llm,
        tenant_id=args.tenant,
        department_id=args.department
    )
        
    print("\nChatbot loaded! Type 'exit' or 'quit' to end the session.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
                
            response = chatbot.get_response(user_input)
            print(f"\nAssistant: {response}\n")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error] Chat turn failed: {e}\n")

if __name__ == "__main__":
    main()
