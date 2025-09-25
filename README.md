
# AmpD Enertainer Chatbot

This application is an interactive chatbot for querying the AmpD Enertainer User Manual. It supports text and image retrieval from the manual PDF, allowing users to ask questions and view relevant content and images.

## Project Structure

```
Ampd_RAG_Application/
│
├── AmpdEnergy_doc_chatbot/
│   ├── agent.py
│   ├── chatbot.py
│   ├── retriever.py
│   ├── generator.py
│   ├── create_index.py
│   ├── requirements.txt
│   ├── meta.json
│   ├── index.faiss
│   ├── feedback.csv
│   ├── logo.png
│   ├── extracted_images/
│   ├── UI_Images/
│   ├── utils/
│  
│
└── README.md
```

## Features
- Ask questions about the AmpD Enertainer manual
- Retrieves relevant text and images from the PDF
- Displays sources and images in the chat UI
- Feedback system for user responses

## Setup Instructions

### 1. Prerequisites
- Python 3.8+
- pip
- Chrome or any modern browser
- Get your Gemini API key from  https://aistudio.google.com/ for free and place it .env file
   then only application will work

### 2. Clone the Repository
```
git clone <your-repo-url>
cd AIAgents_AgenticAI_AgenticRAG_MCP_A2A/AgenticRAGs/AmpdEnergy_doc_chatbot
```


### 3. Create and Activate Virtual Environment
It is recommended to use a Python virtual environment to avoid dependency conflicts.

I already created venv for time saving you just activate it. if require we can keep it in .gitignore

On Windows (PowerShell):
```
python -m venv venv
.\venv\Scripts\activate
```

On macOS/Linux:
```
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```
pip install -r requirements.txt
```

#### Additional Requirements
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) must be installed for image text extraction.
    - On Windows, download and install from https://github.com/tesseract-ocr/tesseract
    - Update the path in `pdf_utils.py` if needed:
      ```python
      pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
      ```

### 4. Prepare the PDF Manual
- Place your PDF manual in the project directory (default: `Ampd Enertainer User Manual (NCM) - Rev 2.3.pdf`).
- For security reasons I removed PDF 

### 5. Build the Index
Run the indexing script to extract text and images and build the FAISS index:
```
python create_index.py --doc "Ampd Enertainer User Manual (NCM) - Rev 2.3.pdf" --index index.faiss --meta meta.json
```
- This will create `index.faiss` and `meta.json` files, and extract images to `extracted_images/`.

- Note: You no need run this file becase already data is the in Vectore db , you can directly run streamlit app. If you can run after cleaning db and meta.json file and remove images well for preventing duplicate data.

### 6. Run the Chatbot
Start the Streamlit app:
```
streamlit run chatbot.py
```
- Open the provided local URL in your browser (usually http://localhost:8501)

## Usage
- Type your question in the input box and click "Send".
- The chatbot will retrieve relevant text and images from the manual.
- Sources and images are shown in expandable sections.
- Use the feedback buttons to rate the answers.

## Troubleshooting
- If images are not displayed, ensure `extracted_images/` contains the PNG files and paths in `meta.json` are correct.
- If Tesseract OCR is not found, check the path in `pdf_utils.py`.
- If Streamlit is not installed, run `pip install streamlit`.
- For embedding or FAISS errors, ensure all dependencies in `requirements.txt` are installed.

## Customization
- To use a different PDF, update the `--doc` argument in the indexing step.
- You can adjust chunk size and overlap in `create_index.py`.
- The chatbot UI and logic can be customized in `chatbot.py`.

## Feedback & Support
- For issues or suggestions, contact support@ampd.energy or use the feedback buttons in the app.

---

**Enjoy querying your AmpD Enertainer manual with images and smart search!**
