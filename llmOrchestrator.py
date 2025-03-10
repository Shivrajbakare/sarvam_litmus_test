import time
from openai import OpenAI
from anthropic import Anthropic
from collections import defaultdict

class CircuitBreaker:
    def __init__(self, failure_threshold=3, reset_timeout=30):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = defaultdict(int)
        self.last_failure_time = {}

    def record_success(self, provider):
        self.failures[provider] = 0

    def record_failure(self, provider):
        self.failures[provider] += 1
        self.last_failure_time[provider] = time.time()

    def is_open(self, provider):
        current_failures = self.failures[provider]
        if current_failures < self.failure_threshold:
            return False
            
        elapsed = time.time() - self.last_failure_time.get(provider, 0)
        if elapsed > self.reset_timeout:
            self.failures[provider] = 0
            return False
        return True

class LLMOrchestrator:
    def __init__(self, config):
        self.config = config or {}
        self.providers = []
        self.circuit_breaker = CircuitBreaker()
        self.setup_providers()

    def setup_providers(self):
        openai_key = self.config.get("OPENAI_API_KEY", "").strip()
        anthropic_key = self.config.get("ANTHROPIC_API_KEY", "").strip()

        if openai_key:
            self.add_openai_provider(openai_key)
        if anthropic_key:
            self.add_anthropic_provider(anthropic_key)

        self.providers.sort(key=lambda p: p["priority"])

    def add_openai_provider(self, api_key):
        client = OpenAI(api_key=api_key)
        self.providers.append({
            "name": "openai",
            "priority": 1,
            "client": client,
            "is_available": lambda: not self.circuit_breaker.is_open("openai"),
            "generate": self._openai_generate,
            "stream": self._openai_stream
        })

    def add_anthropic_provider(self, api_key):
        client = Anthropic(api_key=api_key)
        self.providers.append({
            "name": "anthropic",
            "priority": 2,
            "client": client,
            "is_available": lambda: not self.circuit_breaker.is_open("anthropic"),
            "generate": self._anthropic_generate,
            "stream": self._anthropic_stream
        })

    def _openai_generate(self, prompt, options):
        if not prompt.strip():
            return "I'm ready to help! What would you like to know?"

        try:
            response = self.providers[0]["client"].chat.completions.create(
                model=options.get("model_name", "gpt-3.5-turbo"),
                messages=[{"role": "user", "content": prompt}],
                temperature=min(max(options.get("temperature", 0.7), 0), 1),
                max_tokens=min(options.get("max_tokens", 1000), 4000)
            )
            self.circuit_breaker.record_success("openai")
            return response.choices[0].message.content
        except Exception:
            self.circuit_breaker.record_failure("openai")
            return self._try_next_provider(prompt, options)

    def _anthropic_generate(self, prompt, options):
        if not prompt.strip():
            return "I'm here to assist! What's on your mind?"

        try:
            response = self.providers[1]["client"].messages.create(
                model=options.get("model_name", "claude-2"),
                max_tokens=min(options.get("max_tokens", 1000), 4000),
                temperature=min(max(options.get("temperature", 0.7), 0), 1),
                messages=[{"role": "user", "content": prompt}]
            )
            self.circuit_breaker.record_success("anthropic")
            return response.content[0].text
        except Exception:
            self.circuit_breaker.record_failure("anthropic")
            return self._try_next_provider(prompt, options)

    def _try_next_provider(self, prompt, options):
        available_providers = [p for p in self.providers if p["is_available"]()]
        for provider in available_providers:
            result = provider["generate"](prompt, options)
            if result:
                return result
        return "I'll be ready to help you shortly! Please try again in a moment."

    def generate_response(self, prompt, model_name="gpt-3.5-turbo", temperature=0.7):
        if not self.providers:
            return "I'm getting ready to help! Please ensure your API keys are configured."

        options = {
            "model_name": model_name,
            "temperature": temperature
        }
        
        return self._try_next_provider(prompt, options)

    def stream_response(self, prompt, model_name="gpt-3.5-turbo", temperature=0.7):
        available_providers = [p for p in self.providers if p["is_available"]()]
        for provider in available_providers:
            yield from provider["stream"](prompt, {
                "model_name": model_name,
                "temperature": temperature
            })
