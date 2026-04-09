from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

def test_ollama():
    """
    Tests the local Ollama installation and the 'gemma4:e4b' model.
    """
    print("--- Ollama Connection Test ---")
    
    # Initialize ChatOllama with the specified model
    # By default, it connects to http://localhost:11434
    llm = ChatOllama(model="gemma4:e4b")
    
    print(f"Target Model: gemma4:e4b")
    print("Testing connection and generating response...")
    
    try:
        # Simple test prompt
        messages = [
            HumanMessage(content="Explain the 'E' in 'E4B' regarding the Gemma model in one sentence.")
        ]
        
        # Invoke the model
        response = llm.invoke(messages)
        
        print("\n[SUCCESS] Response received:")
        print("-" * 30)
        print(response.content)
        print("-" * 30)
        
    except Exception as e:
        print("\n[ERROR] Failed to communicate with Ollama.")
        print(f"Details: {e}")
        print("\nTroubleshooting steps:")
        print("1. Ensure Ollama Desktop is running.")
        print("2. Verify the model is downloaded by running: ollama pull gemma4:e4b")
        print("3. Check if the Ollama server is accessible at http://localhost:11434")

if __name__ == "__main__":
    test_ollama()
