from prompt_manager import PromptManager, SimpleUserRepo, SimpleCache, SimpleLogger
import asyncio

async def run_demo():
    # Set up the components
    user_repo = SimpleUserRepo()
    cache = SimpleCache()
    logger = SimpleLogger()
    
    # Initialize the prompt manager
    prompt_manager = PromptManager(user_repo, cache, logger)
    
    # Example conversation
    messages = [
        {"role": "user", "content": "Tell me about quantum physics"},
        {"role": "assistant", "content": "Quantum physics studies atomic behavior"},
        {"role": "user", "content": "Make it simpler"}
    ]
    
    # Test all features
    prompt = await prompt_manager.create_prompt(messages)
    system_message = await prompt_manager.get_system_message("user123")
    template = prompt_manager.get_prompt_template("qa")
    filled_template = prompt_manager.fill_template(template, {"question": "What is coding?"})
    
    # Display results
    print("Generated Prompt:", prompt)
    print("\nSystem Message:", system_message)
    print("\nFilled Template:", filled_template)

if __name__ == "__main__":
    asyncio.run(run_demo())
