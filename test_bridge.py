import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

def main():
    # Load .env from configs directory
    env_path = Path(__file__).parent / 'configs' / '.env'
    load_dotenv(dotenv_path=env_path)

    # Verify API Key exists
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_google_ai_studio_key_here":
        print("Error: GEMINI_API_KEY is not set or is using the placeholder.")
        print(f"Please configure it in {env_path.resolve()}")
        return

    try:
        # Initialize the ChatGoogleGenerativeAI model
        # Using gemini-pro as requested for testing
        llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=api_key)

        # Prepare the prompt
        prompt = "As a strict risk manager, summarize the importance of the VIX index in swing trading in 50 words."
        print(f"\nSending Prompt: '{prompt}'\n")
        
        # Send the prompt
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Output the response
        print("Response:\n--------------------------")
        print(response.text)

    except Exception as e:
        print(f"\nAn error occurred during API call: {e}")

if __name__ == "__main__":
    main()
