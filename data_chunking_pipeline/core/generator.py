import os
from openai import OpenAI

class LLMGenerator:
    """
    Component to generate answers using retrieved context.
    Supports OpenAI API (cloud) and Hugging Face Transformers (local).
    """
    def __init__(self, config, openai_api_key=None, local_model_name=None):
        self.config = config
        self.provider = config.get("llm_provider", "none").lower()
        
        if self.provider == "openai":
            # Initialize OpenAI Client
            api_key = openai_api_key or config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key must be provided either via CLI, config, or OPENAI_API_KEY env variable.")
            self.client = OpenAI(api_key=api_key)
            self.model_name = config.get("openai_model_name", "gpt-4o-mini")
            print(f"[Generator] OpenAI client initialized with model: {self.model_name}")
            
        elif self.provider == "local":
            import transformers
            import torch
            
            # Initialize local HuggingFace model
            self.model_name = local_model_name or config.get("local_model_name", "microsoft/Phi-3-mini-4k-instruct")
            device = config.get("local_device", "auto")
            
            # Set torch dtype based on cuda availability
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            print(f"[Generator] Loading local Hugging Face model: {self.model_name} on device: {device}...")
            
            # Load tokenizer and model pipeline
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.tokenizer,
                torch_dtype=torch_dtype,
                device_map=device
            )
            print("[Generator] Local model loaded successfully.")
            
        else:
            print("[Generator] Provider is 'none'. Generation disabled.")

    def generate_answer(self, query, context_str):
        """
        Generates an answer to the query based on the context.
        
        Args:
            query (str): User's query.
            context_str (str): Formatted context block.
            
        Returns:
            str: Generated answer.
        """
        if self.provider == "none":
            return "Generation disabled (llm_provider set to 'none')."
            
        system_instruction = (
            "You are a precise technical assistant. Use the provided context below to answer the user query.\n"
            "Each document in the context starts with 'Document [number]: Source: <file> (Page <page>)'.\n"
            "You MUST cite the source of your information by appending the corresponding bracketed document number at the end of the sentence containing the cited facts (e.g. '[1]', '[2]').\n"
            "If the answer cannot be found in the context, state that you do not know. "
            "Do not make up facts or use outside knowledge. Keep the response accurate, factual, and correctly cited based ONLY on the context.\n\n"
            f"Context:\n{context_str}"
        )
        
        if self.provider == "openai":
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": query}
                    ],
                    temperature=0.0
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error during OpenAI API generation: {e}"
                
        elif self.provider == "local":
            try:
                messages = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": query}
                ]
                
                # Format using tokenizer chat template if available
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                except Exception:
                    # Fallback to simple instruction string formatting
                    prompt = f"System:\n{system_instruction}\n\nUser:\n{query}\n\nAssistant:\n"
                
                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=512,
                    do_sample=False, # Greedier and more factual
                    temperature=0.0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                generated_text = outputs[0]["generated_text"]
                
                # Extract assistant reply from generated text
                if prompt in generated_text:
                    reply = generated_text[len(prompt):]
                else:
                    reply = generated_text
                return reply.strip()
                
            except Exception as e:
                return f"Error during local model generation: {e}"
                
        return "Unknown provider configuration."

    def generate_raw(self, system_prompt, user_prompt):
        """
        Generates a direct completion without RAG constraints.
        """
        if self.provider == "none":
            return "Generation disabled."
            
        if self.provider == "openai":
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error during raw OpenAI generation: {e}"
                
        elif self.provider == "local":
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                except Exception:
                    prompt = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"
                
                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                generated_text = outputs[0]["generated_text"]
                if prompt in generated_text:
                    reply = generated_text[len(prompt):]
                else:
                    reply = generated_text
                return reply.strip()
            except Exception as e:
                return f"Error during raw local generation: {e}"
                
        return "Unknown provider configuration."
