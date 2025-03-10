import asyncio
from typing import List, Dict, Optional

class PromptManager:
    def __init__(self, user_repo, cache, logger):
        self.user_repo = user_repo
        self.cache = cache
        self.logger = logger
        
    async def create_prompt(self, messages: List[Dict], additional_context: str = "") -> str:
        prompt = "System: You are a helpful AI assistant. Respond accurately and helpfully.\n\n"
        
        for message in messages:
            if message.get("role") == "system":
                prompt = f"System: {message.get('content', '')}\n\n"
                break
                
        if additional_context:
            prompt += f"Context:\n{additional_context}\n\n"
            
        chat_messages = [m for m in messages if m.get("role") != "system"][-10:]
        
        for msg in chat_messages:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            prompt += f"{role}: {content}\n\n"
            
        prompt += "Assistant: "
        return prompt

    async def get_system_message(self, user_id: str) -> str:
        cache_key = f"system_msg:{user_id}"
        
        cached_message = await self.cache.get(cache_key)
        if cached_message:
            return cached_message
            
        system_message = "You are a helpful AI assistant. "
        
        user_prefs = await self.user_repo.get_user_preferences(user_id) or {}
        
        if personality := user_prefs.get("personality"):
            system_message += f"Your personality is {personality}. "
            
        if expertise := user_prefs.get("expertise", []):
            system_message += f"You specialize in {', '.join(expertise)}. "
            
        if style := user_prefs.get("response_style"):
            system_message += f"Your responses should be {style}. "
            
        await self.cache.set(cache_key, system_message, 3600)
        return system_message

    def get_prompt_template(self, template_name: str) -> Optional[str]:
        templates = {
            "qa": "Answer the following question accurately and concisely: {{question}}",
            "summarize": "Provide a clear summary of the following text: {{text}}",
            "creative": "Generate a creative {{format}} about {{topic}}",
            "code": "Write {{language}} code that accomplishes the following: {{task}}"
        }
        return templates.get(template_name)

    def fill_template(self, template: Optional[str], variables: Dict) -> Optional[str]:
        if not template:
            return None
            
        filled = template
        for key, value in variables.items():
            filled = filled.replace(f"{{{{{key}}}}}", str(value))
        return filled

class SimpleUserRepo:
    async def get_user_preferences(self, user_id: str) -> Dict:
        return {
            "personality": "friendly",
            "expertise": ["AI", "Mathematics"],
            "response_style": "detailed",
        }

class SimpleCache:
    def __init__(self):
        self.cache = {}
        
    async def get(self, key: str) -> Optional[str]:
        return self.cache.get(key)
        
    async def set(self, key: str, value: str, ttl: int) -> None:
        self.cache[key] = value

class SimpleLogger:
    def error(self, message: str, data: Dict) -> None:
        print(f"LOG: {message} | {data}")
