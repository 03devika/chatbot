# main_chatbot.py - Complete chatbot with all features - FIXED VERSION
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import PyPDF2
import tempfile
import os

class SmartChatbot:
    def __init__(self):
        """Initialize the chatbot - this runs when you create a new chatbot"""
        print("🤖 Initializing chatbot...")
        self.setup_components()
        print("✅ Chatbot ready!")
    
    def setup_components(self):
        """Set up all the AI components"""
        
        # 1. Initialize the language model (the "brain")
        print("📡 Connecting to Ollama...")
        self.llm = OllamaLLM(
            model="gemma2:2b",   # Your chosen Gemma2 model
            temperature=0.7      # How creative the responses are (0-1)
        )
        
        # 2. Initialize embeddings (converts text to numbers for searching)
        print("🔢 Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"  # Good balance of speed and quality
        )
        
        # 3. Text splitter (breaks large documents into smaller chunks)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,     # Optimized for Gemma2:2b
            chunk_overlap=50,   # Less overlap needed for small model
            separators=["\n\n", "\n", ". ", " ", ""]  # Where to split text
        )
        
        # 4. Memory (remembers conversation history) - FIXED
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # This fixes the multiple output keys error
        )
        
        # 5. Default system prompt (optimized for Gemma2)
        self.system_prompt = "You are a helpful assistant. Give clear, direct answers. Be concise but informative."
        
        print("✅ All components loaded!")
    
    def process_pdf(self, uploaded_file):
        """Process a PDF file and make it searchable"""
        print(f"📄 Processing PDF: {uploaded_file.name}")
        
        # Step 1: Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Step 2: Extract text from PDF
        text = ""
        try:
            with open(tmp_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                print(f"📖 Reading {len(pdf_reader.pages)} pages...")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += page_text
                    print(f"   Page {page_num + 1}: {len(page_text)} characters")
        
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            return f"Error processing PDF: {e}"
        
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
        
        # Step 3: Split text into chunks
        print("✂️ Splitting text into chunks...")
        chunks = self.text_splitter.split_text(text)
        print(f"📚 Created {len(chunks)} chunks")
        
        # Step 4: Create vector database (searchable knowledge base)
        print("🧠 Creating knowledge base...")
        self.vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"  # Saves to disk so you don't lose it
        )
        
        # Step 5: Create question-answering chain - FIXED
        print("🔗 Setting up Q&A system...")
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 3}  # Find 3 most relevant chunks
            ),
            memory=self.memory,
            return_source_documents=True,  # Show which parts of document were used
            verbose=True  # Helpful for debugging
        )
        
        return f"✅ Successfully processed PDF! Created {len(chunks)} searchable chunks."
    
    def set_system_prompt(self, prompt):
        """Change how the chatbot behaves"""
        self.system_prompt = prompt
        print(f"🎭 System prompt updated: {prompt[:50]}...")
    
    def chat(self, user_input):
        """Generate a response to user input"""
        print(f"💬 User asked: {user_input}")
        
        try:
            if hasattr(self, 'qa_chain'):
                # Use document-aware conversation
                print("🔍 Searching documents for relevant information...")
                response = self.qa_chain.invoke({"question": user_input})
                answer = response['answer']
                
                # Show which documents were used (optional)
                if 'source_documents' in response and response['source_documents']:
                    print(f"📄 Used {len(response['source_documents'])} document chunks")
                    # Optionally, you can add source information to the answer
                    # answer += f"\n\n*Based on {len(response['source_documents'])} relevant sections from the uploaded document.*"
                
                return answer
            else:
                # Regular conversation without documents
                print("💭 Generating regular response...")
                full_prompt = f"{self.system_prompt}\n\nHuman: {user_input}\nAssistant:"
                return self.llm.invoke(full_prompt)
        
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"Sorry, I encountered an error: {e}"

# Streamlit Web Interface
def main():
    """Main function that creates the web interface"""
    
    # Page configuration
    st.set_page_config(
        page_title="My Smart Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("Chatbot")
    st.write("Upload PDFs, ask questions, and chat with AI!")
    
    # Initialize chatbot (only once)
    if 'chatbot' not in st.session_state:
        with st.spinner("🚀 Loading chatbot..."):
            st.session_state.chatbot = SmartChatbot()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # System prompt section
        st.subheader("🎭 Personality")
        system_prompt = st.text_area(
            "How should the chatbot behave?",
            value="You are a helpful assistant analyzing sales data. Provide specific insights and numbers when available. Be analytical and data-focused.",
            height=100,
            help="This controls how the chatbot responds"
        )
        
        if st.button("Update Personality"):
            st.session_state.chatbot.set_system_prompt(system_prompt)
            st.success("✅ Personality updated!")
        
        st.divider()
        
        # PDF upload section
        st.subheader("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload a PDF to chat about",
            type="pdf",
            help="The chatbot will read this document and answer questions about it"
        )
        
        if uploaded_file:
            st.write(f"Selected: {uploaded_file.name}")
            if st.button("🔄 Process PDF"):
                with st.spinner("📖 Reading and processing PDF..."):
                    result = st.session_state.chatbot.process_pdf(uploaded_file)
                    st.success(result)
        
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.chatbot.memory.clear()
            st.success("Conversation cleared!")
    
    # Main chat interface
    st.header("💬 Chat")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your sales data..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your data..."):
                response = st.session_state.chatbot.chat(prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# This runs when you execute the file
if __name__ == "__main__":
    main()