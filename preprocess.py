"""
Federal Board Study Bot - PDF Preprocessing Script
This script processes PDF files for each Grade+Subject combination and creates ChromaDB embeddings.

Directory structure expected:
data/
  grade_9/
    mathematics.pdf
    biology.pdf
    chemistry.pdf
    physics.pdf
    computer_science.pdf
  grade_10/
    ...
  grade_11/
    ...
  grade_12/
    ...

Run this script once to preprocess all PDFs and create embeddings.
"""

import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import tiktoken
import pytesseract
import fitz  
from PIL import Image
import io

# Load environment variables
load_dotenv(override=True)

# Configuration
GRADES = ['grade_9', 'grade_10', 'grade_11', 'grade_12']
SUBJECTS = ['mathematics', 'biology', 'chemistry', 'physics', 'computer_science']
DATA_DIR = Path('data')
EMBEDDINGS_DIR = Path('embeddings')
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Configure Tesseract path for Windows
# Update this path if Tesseract is installed elsewhere
TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', '')),
]

# Try to find Tesseract
for path in TESSERACT_PATHS:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        print(f"Found Tesseract at: {path}")
        break
else:
    print("Warning: Tesseract OCR not found. Please install it from:")
    print("https://github.com/UB-Mannheim/tesseract/wiki")
    print("Or set the correct path in the TESSERACT_PATHS list")

def count_tokens(text, model="gpt-3.5-turbo"):
    """Count tokens in text"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def extract_text_with_ocr(pdf_path, max_pages=None):
    """Extract text from PDF using OCR for image-based pages (PyMuPDF version)"""
    print(f"Extracting text with OCR from: {pdf_path}")
    
    try:
        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        pages_to_process = min(max_pages, total_pages) if max_pages else total_pages
        
        print(f"PDF has {total_pages} pages, processing {pages_to_process}")
        
        documents = []
        
        for page_num in range(pages_to_process):
            print(f"Processing page {page_num + 1}/{pages_to_process} with OCR...")
            
            # Get page
            page = doc[page_num]
            
            # Try different zoom levels for better OCR
            text_extracted = False
            for zoom in [1.5, 2.0, 3.0]:
                # Convert page to image with different zoom
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(img_data))
                
                # Try different OCR configurations
                ocr_configs = ['--psm 6', '--psm 3', '--psm 1']
                
                for config in ocr_configs:
                    try:
                        # Extract text using OCR
                        text = pytesseract.image_to_string(image, lang='eng', config=config)
                        
                        if text.strip() and len(text.strip()) > 10:  # More than 10 characters
                            # Create a document-like object
                            from langchain.schema import Document
                            doc_obj = Document(
                                page_content=text,
                                metadata={
                                    'page': page_num,
                                    'source': str(pdf_path),
                                    'zoom': zoom,
                                    'config': config
                                }
                            )
                            documents.append(doc_obj)
                            text_extracted = True
                            break  # Success, move to next page
                    except Exception as e:
                        continue  # Try next config
                
                if text_extracted:
                    break  # Success, move to next page
            
            if not text_extracted:
                print(f"No text extracted from page {page_num + 1}")
        
        doc.close()
        return documents
        
    except Exception as e:
        print(f"Error in OCR processing: {str(e)}")
        return []

def load_and_split_pdf(pdf_path):
    """Load PDF and split into chunks, using OCR for all PDFs"""
    print(f"Loading PDF: {pdf_path}")
    
    # Force OCR for all PDFs since they are scanned images
    print(f"Using OCR for all PDFs (scanned images)...")
    pages = extract_text_with_ocr(pdf_path)
    
    if not pages:
        print(f"Warning: No content extracted from {pdf_path}")
        return []
    
    print(f"Loaded {len(pages)} pages from {pdf_path}")
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=count_tokens,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Split documents and add metadata
    chunks = []
    for page in pages:
        # Check if page has content
        if hasattr(page, 'page_content') and page.page_content.strip():
            page_chunks = text_splitter.split_documents([page])
            for chunk in page_chunks:
                # Add additional metadata
                chunk.metadata.update({
                    'file_path': str(pdf_path),
                    'page_number': page.metadata.get('page', 0) + 1,  # Make 1-indexed
                    'chapter': extract_chapter_info(chunk.page_content)
                })
            chunks.extend(page_chunks)
        else:
            print(f"Skipping empty page {page.metadata.get('page', 0) + 1}")
    
    print(f"Created {len(chunks)} chunks from {pdf_path}")
    return chunks

def extract_chapter_info(text):
    """Extract chapter information from text content"""
    lines = text.split('\n')
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip().lower()
        if 'chapter' in line or 'unit' in line:
            return line.title()
    
    # Look for numbered sections or headers
    for line in lines[:10]:
        line = line.strip()
        if len(line) < 100 and any(char.isdigit() for char in line[:10]):
            if any(keyword in line.lower() for keyword in ['introduction', 'definition', 'theorem', 'example']):
                return line[:50] + "..." if len(line) > 50 else line
    
    return "General Content"

def create_vector_store(documents, grade, subject):
    """Create ChromaDB vector store for grade+subject combination"""
    if not documents:
        print(f"No documents to process for {grade}_{subject}")
        return
    
    print(f"Creating vector store for {grade}_{subject}...")
    
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Create directory for this grade+subject
    db_path = EMBEDDINGS_DIR / f"{grade}_{subject}"
    db_path.mkdir(parents=True, exist_ok=True)
    
    # Create ChromaDB vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(db_path),
        collection_name=f"{grade}_{subject}_collection"
    )
    
    # Persist the database
    vectorstore.persist()
    
    print(f"Vector store created and saved to {db_path}")
    print(f"Total documents in store: {len(documents)}")

def validate_pdf_structure():
    """Validate that the expected PDF files exist"""
    missing_files = []
    
    for grade in GRADES:
        grade_dir = DATA_DIR / grade
        if not grade_dir.exists():
            print(f"Warning: Directory {grade_dir} does not exist")
            continue
            
        for subject in SUBJECTS:
            pdf_file = grade_dir / f"{subject}.pdf"
            if not pdf_file.exists():
                missing_files.append(str(pdf_file))
    
    if missing_files:
        print("Missing PDF files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure all required PDFs are in place before running preprocessing.")
        return False
    
    return True

def main():
    """Main preprocessing function"""
    print("=== Federal Board Study Bot - PDF Preprocessing ===")
    print()
    
    # Check if Google API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please set your Google Gemini API key in the .env file.")
        return
    
    # Validate PDF structure
    print("Validating PDF file structure...")
    if not validate_pdf_structure():
        return
    
    # Create embeddings directory
    EMBEDDINGS_DIR.mkdir(exist_ok=True)
    
    # Process each grade and subject
    total_processed = 0
    successful_processing = []
    
    for grade in GRADES:
        grade_dir = DATA_DIR / grade
        
        if not grade_dir.exists():
            print(f"Skipping {grade} - directory not found")
            continue
            
        for subject in SUBJECTS:
            pdf_file = grade_dir / f"{subject}.pdf"
            
            if not pdf_file.exists():
                print(f"Skipping {grade}_{subject} - PDF not found")
                continue
            
            try:
                print(f"\n--- Processing {grade}_{subject} ---")
                
                # Load and split PDF
                documents = load_and_split_pdf(pdf_file)
                
                if documents:
                    # Create vector store
                    create_vector_store(documents, grade, subject)
                    successful_processing.append(f"{grade}_{subject}")
                    total_processed += len(documents)
                else:
                    print(f"No content extracted from {pdf_file}")
                    
            except Exception as e:
                print(f"Error processing {grade}_{subject}: {str(e)}")
                continue
    
    # Print summary
    print(f"\n=== Preprocessing Complete ===")
    print(f"Total documents processed: {total_processed}")
    print(f"Successfully processed combinations:")
    for combo in successful_processing:
        print(f"  - {combo}")
    
    if successful_processing:
        print(f"\nVector stores saved in: {EMBEDDINGS_DIR}")
        print("You can now run the Streamlit app with: streamlit run app.py")
    else:
        print("No files were successfully processed. Please check your PDF files and try again.")

if __name__ == "__main__":
    main()
