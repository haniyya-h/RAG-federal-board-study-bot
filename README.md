# 📚 Federal Board Study Bot

An AI-powered study companion for Federal Board students in Pakistan, built with RAG (Retrieval-Augmented Generation) technology. This chatbot helps students understand their textbook content by answering questions directly from official PDFs and providing relevant practice questions.

## 🌟 What It Is

The Federal Board Study Bot is a smart study assistant that:
- **Answers questions** from official Federal Board textbooks (Grades 9-12)
- **Provides page references** and chapter information
- **Generates practice questions** tailored to what students are studying
- **Supports all subjects**: Mathematics, Biology, Chemistry, Physics, Computer Science
- **Uses OCR technology** to read scanned textbook images
- **Gives student-friendly explanations** in simple, encouraging language

## 🚀 How It Works

### 1. **Document Processing**
- PDFs are processed using OCR (Optical Character Recognition) to extract text from scanned images
- Text is split into chunks and converted to embeddings using Google's Gemini AI
- Vector database (ChromaDB) stores embeddings for fast retrieval

### 2. **Question Answering**
- When students ask questions, the system searches the vector database for relevant content
- Google Gemini AI generates answers based on the retrieved textbook content
- Answers include page numbers and chapter references

### 3. **Practice Questions**
- AI generates 3 relevant SLO (Student Learning Outcome) style questions
- Questions are tailored to the specific topic and grade level
- Designed to help students test their understanding

## 🛠️ Tech Stack

### **Core Technologies**
- **Python 3.12+** - Main programming language
- **Streamlit** - Web application framework
- **LangChain** - LLM application framework
- **ChromaDB** - Vector database for embeddings
- **Google Gemini AI** - LLM and embeddings provider

### **OCR & Document Processing**
- **PyMuPDF (fitz)** - PDF processing and image extraction
- **PyTesseract** - OCR engine for text extraction
- **Pillow (PIL)** - Image manipulation

### **Additional Libraries**
- **python-dotenv** - Environment variable management
- **pypdf** - PDF text extraction
- **tiktoken** - Token counting
- **numpy** - Numerical operations

## 📋 Prerequisites

Before running the application, ensure you have:

1. **Python 3.12+** installed
2. **Tesseract OCR** installed on your system
3. **Google Gemini API key** (free tier available)

### Installing Tesseract OCR

**Windows:**
```bash
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Or use chocolatey:
choco install tesseract
```

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
```

## 🚀 Quick Start

### 1. **Clone the Repository**
```bash
git clone <repository-url>
cd RAG
```

### 2. **Set Up Virtual Environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4. **Set Up Environment Variables**
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get your free API key from [Google AI Studio](https://aistudio.google.com/)

### 5. **Add Your Textbooks (Optional)**
**🎉 Good News:** The repository includes pre-processed embeddings, so you can use it immediately!

**To add your own textbooks:**
1. Place your Federal Board PDF textbooks in the `data` folder:
```
data/
├── grade_9/
│   ├── mathematics.pdf
│   ├── biology.pdf
│   ├── chemistry.pdf
│   ├── physics.pdf
│   └── computer_science.pdf
├── grade_10/
│   └── ... (same structure)
├── grade_11/
│   └── ... (same structure)
└── grade_12/
    └── ... (same structure)
```
2. Run preprocessing: `python preprocess.py`

**📚 Where to get PDFs:**
- Official Federal Board website
- Your school's digital library
- Educational resource websites

### 6. **Process the Textbooks (Optional)**
The repository includes pre-processed embeddings for immediate use! If you want to add your own textbooks:

```bash
python preprocess.py
```
This will:
- Extract text from PDFs using OCR
- Create embeddings using Google Gemini
- Store data in ChromaDB vector database

### 7. **Run the Application**
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## 🌐 Deployment Options

### **Option 1: Streamlit Cloud (Recommended)**

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Add your `GOOGLE_API_KEY` in the secrets section
   - Deploy!

3. **Add Secrets**
   In Streamlit Cloud, add:
   ```
   GOOGLE_API_KEY = "your_api_key_here"
   ```

### **Option 2: Heroku**

1. **Create Procfile**
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Add runtime.txt**
   ```
   python-3.12.0
   ```

3. **Deploy**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

### **Option 3: Docker**

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.12-slim

   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt

   # Install Tesseract
   RUN apt-get update && apt-get install -y tesseract-ocr

   COPY . .
   EXPOSE 8501

   CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
   ```

2. **Build and Run**
   ```bash
   docker build -t federal-board-bot .
   docker run -p 8501:8501 -e GOOGLE_API_KEY=your_key federal-board-bot
   ```

### **Option 4: Use Pre-deployed Version**

If you don't want to deploy yourself, you can use the live version at:
**🔗 [Federal Board Study Bot](https://your-deployed-url.com)**

Simply:
1. Select your grade and subject
2. Ask any question about your textbook
3. Get instant answers with practice questions!

## 📁 Project Structure

```
RAG/
├── app.py                 # Main Streamlit application
├── preprocess.py          # PDF processing and embedding creation
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── slo_data.json         # SLO questions data (optional)
├── data/                 # PDF textbooks directory
│   ├── grade_9/
│   ├── grade_10/
│   ├── grade_11/
│   └── grade_12/
├── embeddings/           # ChromaDB vector stores
│   ├── grade_9_mathematics/
│   ├── grade_9_biology/
│   └── ...
└── README.md            # This file
```

## 🎯 Features

### **For Students**
- ✅ Ask questions in natural language
- ✅ Get answers directly from official textbooks
- ✅ See page numbers and chapter references
- ✅ Receive practice questions for self-testing
- ✅ Student-friendly, encouraging explanations
- ✅ Support for all grades (9-12) and subjects

### **For Educators**
- ✅ Easy to add new textbooks
- ✅ Customizable for different curricula
- ✅ Tracks student interactions
- ✅ Generates relevant practice questions

## 🔧 Configuration

### **Customizing the Bot**
- **Chunk Size**: Modify `CHUNK_SIZE` in `preprocess.py` (default: 800)
- **Chunk Overlap**: Modify `CHUNK_OVERLAP` in `preprocess.py` (default: 100)
- **Answer Style**: Edit the prompt template in `create_custom_prompt()`
- **SLO Questions**: Modify the generation prompt in `generate_slo_questions()`

### **Adding New Subjects**
1. Add subject to `SUBJECTS` list in both `app.py` and `preprocess.py`
2. Place PDF in appropriate grade folder
3. Run preprocessing script

## 🐛 Troubleshooting

### **Common Issues**

**"No data found" error:**
- Ensure PDFs are in correct folder structure
- Run `python preprocess.py` to process PDFs

**OCR not working:**
- Install Tesseract OCR on your system
- Check if Tesseract is in your PATH

**API key errors:**
- Verify your Google API key is correct
- Check if you have API quota remaining

**Slow loading:**
- The app caches models after first load
- Subsequent loads will be much faster

## 📊 Performance

- **Initial Load**: ~10-15 seconds (model initialization)
- **Subsequent Loads**: ~2-3 seconds (cached models)
- **Question Answering**: ~3-5 seconds per question
- **Vector Database**: Supports thousands of documents efficiently

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Submit a pull request**

### **Areas for Improvement**
- Support for more file formats
- Better OCR accuracy
- Multi-language support
- Mobile app version
- Integration with learning management systems

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini AI** for providing the language model and embeddings
- **Streamlit** for the amazing web framework
- **LangChain** for the LLM application framework
- **ChromaDB** for vector database capabilities
- **Federal Board of Pakistan** for the educational content

## 📞 Support

If you encounter any issues or have questions:

1. **Check the troubleshooting section** above
2. **Open an issue** on GitHub
3. **Contact the development team**

---

**Made with ❤️ for Federal Board students in Pakistan**

*Empowering students with AI-powered learning tools* 🚀
