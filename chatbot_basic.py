# chatbot_basic.py - Your first chatbot file
print("Testing if Python works...")

# Import required libraries
try:
    from langchain_ollama import OllamaLLM
    print("✅ LangChain Ollama imported successfully")
except ImportError as e:
    print(f"❌ Error importing LangChain Ollama: {e}")
    print("Run: pip install langchain-ollama")

try:
    import streamlit as st
    print("✅ Streamlit imported successfully")
except ImportError as e:
    print(f"❌ Error importing Streamlit: {e}")
    print("Run: pip install streamlit")

# Test Ollama connection
def test_ollama():
    try:
        llm = OllamaLLM(model="gemma2:2b")  # Your chosen model
        response = llm.invoke("Say hello!")
        print(f"✅ Ollama working! Response: {response}")
        return True
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        print("Make sure Ollama is running and model is pulled")
        return False

if __name__ == "__main__":
    print("🚀 Starting chatbot tests...")
    test_ollama()