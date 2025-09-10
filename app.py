"""
Federal Board Study Bot - Streamlit Application
A RAG-based chatbot for Federal Board students using ChromaDB + Gemini
"""

import streamlit as st
import os
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time

# Load environment variables (override=True ensures .env file takes precedence)
load_dotenv(override=True)

# Configuration
GRADES = ['9', '10', '11', '12']
SUBJECTS = ['Mathematics', 'Biology', 'Chemistry', 'Physics', 'Computer Science']
EMBEDDINGS_DIR = Path('embeddings')

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_vectorstore' not in st.session_state:
    st.session_state.current_vectorstore = None
if 'current_grade_subject' not in st.session_state:
    st.session_state.current_grade_subject = None

@st.cache_resource
def load_embeddings():
    """Load Google Generative AI embeddings model with retry logic"""
    try:
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    except Exception as e:
        st.error(f"Failed to load embeddings: {str(e)}")
        st.info("This might be due to API quota limits. Please try again later or check your Google AI Studio quota.")
        return None

@st.cache_resource
def load_llm():
    """Load Google Generative AI language model with error handling"""
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    except Exception as e:
        st.error(f"Failed to load LLM: {str(e)}")
        st.info("This might be due to API quota limits. Please try again later or check your Google AI Studio quota.")
        return None

def subject_to_filename(subject):
    """Convert subject display name to filename format"""
    return subject.lower().replace(' ', '_')

@st.cache_resource
def load_vector_store(grade, subject):
    """Load ChromaDB vector store for specific grade and subject"""
    subject_filename = subject_to_filename(subject)
    db_path = EMBEDDINGS_DIR / f"grade_{grade}_{subject_filename}"
    
    if not db_path.exists():
        return None
    
    try:
        # Use cached embeddings
        embeddings = load_embeddings()
        
        # Load existing ChromaDB
        vectorstore = Chroma(
            persist_directory=str(db_path),
            embedding_function=embeddings,
            collection_name=f"grade_{grade}_{subject_filename}_collection"
        )
        
        return vectorstore
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def generate_slo_questions(question, grade, subject, llm):
    """Generate relevant SLO questions using Gemini based on the student's question"""
    try:
        prompt = f"""You are a helpful study assistant for Federal Board students in Pakistan. 
        A Grade {grade} student studying {subject} asked: "{question}"
        
        Generate exactly 3 relevant SLO (Student Learning Outcome) style practice questions that would help this student understand the topic better.
        Make the questions:
        - Appropriate for Grade {grade} level
        - Related to the topic they asked about
        - In the style of Federal Board exam questions
        - Clear and specific
        - Cover different aspects of the topic if possible
        
        IMPORTANT: Return ONLY the 3 questions, one per line, with no other text, numbering, or formatting.
        Each question must end with a question mark (?).
        
        Example format:
        What is the main function of mitochondria in a cell?
        Explain the process of photosynthesis step by step.
        Calculate the area of a triangle with base 10 cm and height 6 cm."""
        
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Split by newlines and clean up
        questions = []
        for line in content.split('\n'):
            line = line.strip()
            # Remove any numbering or bullet points
            line = line.lstrip('123456789.-•* ')
            if line and '?' in line and len(line) > 10:
                questions.append(line)
        
        if questions:
            print(f"Generated {len(questions)} SLO questions for {subject}")
            return questions[:3]  # Return max 3 questions
        else:
            print(f"No valid SLO questions generated. Raw response: {content[:200]}...")
            return []
            
    except Exception as e:
        print(f"Error generating SLO questions: {e}")
        return []

def create_custom_prompt():
    """Create custom prompt template for the QA chain"""
    template = """You are an academic tutor for Federal Board students in Pakistan. Provide clear, accurate explanations based on the textbook content.

INSTRUCTIONS:
1. Provide clear and concise explanations
2. Use appropriate academic language for the grade level
3. Break down complex concepts into logical steps
4. Include relevant examples when helpful
5. Base your answer strictly on the provided textbook context
6. Maintain a professional yet approachable tone
7. Keep answers focused and under 300 words
8. Ensure accuracy and educational value

CONTEXT from Federal Board Textbook:
{context}

STUDENT'S QUESTION: {question}

ANSWER: """

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def format_response(answer, relevant_docs, slo_questions):
    """Format the complete response with answer, source info, and SLO questions"""
    response = f"**💡 Study Buddy Answer:**\n{answer}\n\n"
    
    # Add source information in a friendly way
    if relevant_docs:
        response += "**📖 Where to Find This in Your Book:**\n"
        pages = set()
        chapters = set()
        
        for doc in relevant_docs:
            if 'page_number' in doc.metadata:
                pages.add(str(doc.metadata['page_number']))
            if 'chapter' in doc.metadata and doc.metadata['chapter'] != "General Content":
                chapters.add(doc.metadata['chapter'])
        
        if pages:
            try:
                sorted_pages = sorted(pages, key=lambda x: int(x) if x.isdigit() else 0)
                response += f"📄 Check page(s): {', '.join(sorted_pages)}\n"
            except (ValueError, TypeError):
                response += f"📄 Check page(s): {', '.join(sorted(pages))}\n"
        if chapters:
            response += f"📚 Chapter: {', '.join(chapters)}\n"
        response += "\n"
    
    # Add SLO questions 
    if slo_questions:
        response += "**🎯 Practice Questions to Test Yourself:**\n"
        for i, question in enumerate(slo_questions, 1):
            response += f"{i}. {question}\n"
        response += "\n💪 *Try these questions to make sure you understand the topic!*\n"
    else:
        response += "**🎯 Practice Questions:**\n"
        response += "*Here are some general practice questions to help you study:*\n"
        response += "1. Can you explain this concept in your own words?\n"
        response += "2. What are the key points you learned from this topic?\n"
        response += "3. How would you apply this knowledge in a real situation?\n"
        response += "\n💪 *Use these questions to test your understanding!*\n"
    
    return response

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Federal Board Study Bot",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Federal Board Study Bot")
    st.markdown("*Your AI-powered study companion for Federal Board textbooks*")
    
    # Show loading indicator while initializing
    if 'app_initialized' not in st.session_state:
        with st.spinner("🚀 Initializing your study buddy..."):
            # Pre-load the embeddings and LLM models
            embeddings = load_embeddings()
            llm = load_llm()
            
            if embeddings is None or llm is None:
                st.error("❌ Failed to initialize AI models. This might be due to API quota limits.")
                st.info("**Solutions:**")
                st.info("1. Wait for quota reset (usually daily)")
                st.info("2. Check your Google AI Studio quota usage")
                st.info("3. Consider upgrading to a paid plan")
                st.info("4. Try again later")
                st.stop()
            
            st.session_state.app_initialized = True
    
    # Check if Google API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("❌ Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
        st.stop()
    
    # Sidebar for grade and subject selection
    st.sidebar.header("📋 Selection")
    
    # Grade selection
    selected_grade = st.sidebar.selectbox("Select Grade:", GRADES, key="grade_select")
    
    # Subject selection
    selected_subject = st.sidebar.selectbox("Select Subject:", SUBJECTS, key="subject_select")
    
    # Check if vector store exists for selected combination
    current_selection = f"grade_{selected_grade}_{subject_to_filename(selected_subject)}"
    
    if current_selection != st.session_state.current_grade_subject:
        # Load new vector store
        with st.spinner("Loading textbook data..."):
            vectorstore = load_vector_store(selected_grade, selected_subject)
            
        if vectorstore is None:
            st.error(f"❌ No data found for Grade {selected_grade} - {selected_subject}")
            st.info("Please run the preprocessing script to generate embeddings for this combination.")
            st.code("python preprocess.py")
            st.stop()
        
        st.session_state.current_vectorstore = vectorstore
        st.session_state.current_grade_subject = current_selection
        st.session_state.chat_history = []  # Clear chat history on subject change
    
    # Display current selection
    st.sidebar.success(f"✅ Grade {selected_grade} - {selected_subject}")
    
    # Main chat interface
    st.header("💬 Ask Your Question")
    
    # Question input
    question = st.text_input(
        "Enter your question about the textbook:",
        placeholder="e.g., What is photosynthesis? or Solve quadratic equations...",
        key="question_input"
    )
    
    # Ask button
    if st.button("Ask Question", type="primary") or question:
        if not question:
            st.warning("Please enter a question.")
            return
        
        if not st.session_state.current_vectorstore:
            st.error("No vector store loaded. Please select a valid grade and subject.")
            return
        
        with st.spinner("Searching textbook and generating answer..."):
            try:
                # Use cached LLM
                llm = load_llm()
                
                # Create custom prompt
                custom_prompt = create_custom_prompt()
                
                # Create retrieval QA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.current_vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 3}
                    ),
                    chain_type_kwargs={"prompt": custom_prompt},
                    return_source_documents=True
                )
                
                # Get answer from textbook
                result = qa_chain({"query": question})
                answer = result["result"]
                source_docs = result["source_documents"]
                
                # Generate relevant SLO questions using Gemini
                slo_questions = generate_slo_questions(
                    question, selected_grade, selected_subject, llm
                )
                
                # If no questions generated, try a simpler approach
                if not slo_questions:
                    print("Trying fallback SLO question generation...")
                    try:
                        simple_prompt = f"Generate 3 simple practice questions for Grade {selected_grade} {selected_subject} about: {question}. Return only the questions, one per line, ending with ?"
                        fallback_response = llm.invoke(simple_prompt)
                        fallback_content = fallback_response.content.strip()
                        fallback_questions = [q.strip().lstrip('123456789.-•* ') for q in fallback_content.split('\n') if q.strip() and '?' in q and len(q.strip()) > 10]
                        if fallback_questions:
                            slo_questions = fallback_questions[:3]
                            print(f"Fallback generated {len(slo_questions)} questions")
                    except Exception as e:
                        print(f"Fallback SLO generation failed: {e}")
                
                # Format complete response
                formatted_response = format_response(answer, source_docs, slo_questions)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": question,
                    "response": formatted_response,
                    "timestamp": time.strftime("%H:%M:%S")
                })
                
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
                return
    
    # Display chat history
    if st.session_state.chat_history:
        st.header("📝 Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {chat['question'][:50]}... ({chat['timestamp']})", expanded=(i==0)):
                st.markdown(chat['response'])
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About")
    st.sidebar.info(
        "This bot uses official Federal Board textbooks to answer your questions. "
        "It also provides relevant SLO-style practice questions for each topic."
    )
    
    st.sidebar.markdown("**Features:**")
    st.sidebar.markdown("• 📖 Textbook-based answers")
    st.sidebar.markdown("• 📄 Page number references")
    st.sidebar.markdown("• 📚 Chapter information")
    st.sidebar.markdown("• 📝 SLO practice questions")
    
    # Clear chat history button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

if __name__ == "__main__":
    main()
