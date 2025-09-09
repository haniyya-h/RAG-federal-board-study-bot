# 📚 Data Directory

This directory contains the Federal Board textbook PDFs organized by grade and subject.

## 📁 Required Structure

```
data/
├── grade_9/
│   ├── mathematics.pdf
│   ├── biology.pdf
│   ├── chemistry.pdf
│   ├── physics.pdf
│   └── computer_science.pdf
├── grade_10/
│   ├── mathematics.pdf
│   ├── biology.pdf
│   ├── chemistry.pdf
│   ├── physics.pdf
│   └── computer_science.pdf
├── grade_11/
│   ├── mathematics.pdf
│   ├── biology.pdf
│   ├── chemistry.pdf
│   ├── physics.pdf
│   └── computer_science.pdf
└── grade_12/
    ├── mathematics.pdf
    ├── biology.pdf
    ├── chemistry.pdf
    ├── physics.pdf
    └── computer_science.pdf
```

## 📋 Instructions

1. **Download PDFs**: Get the official Federal Board textbooks for each grade
2. **Rename Files**: Use the exact filenames shown above (lowercase with underscores)
3. **Place in Folders**: Put each PDF in its corresponding grade folder
4. **Run Preprocessing**: Execute `python preprocess.py` to process the PDFs

## ⚠️ Important Notes

- PDFs should be the official Federal Board textbooks
- Files must be named exactly as shown (case-sensitive)
- All 5 subjects are required for each grade
- PDFs are processed using OCR, so scanned images work fine
- Large PDF files may take time to process

## 🔄 After Adding PDFs

Once you've added all the PDFs:

1. Run the preprocessing script:
   ```bash
   python preprocess.py
   ```

2. Start the application:
   ```bash
   streamlit run app.py
   ```

3. Select your grade and subject to start studying!
