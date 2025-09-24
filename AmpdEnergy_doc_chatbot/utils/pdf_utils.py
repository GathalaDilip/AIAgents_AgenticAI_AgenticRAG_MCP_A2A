# utils/pdf_utils.py
import os
from PIL import Image
import pytesseract
import fitz  # PyMuPDF

# from langchain.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def load_pdf_pages(pdf_path):
    """
    Returns list of langchain Documents (page-level) using PyMuPDF loader.
    """
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load_and_split()  # Each doc corresponds to a page or chunk depending on loader
    # print(f"Loaded {len(docs)} pages from {docs}")
    return docs




def extract_text_and_images(pdf_path, out_dir="extracted_images"):
    """
    Extracts text and images from each page of the PDF.
    Returns a list of dicts per page: {
        "page": page_number,
        "text": page_text,
        "images": [
            {"image_path": ..., "ocr_text": ...}, ...
        ]
    }
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    doc = fitz.open(pdf_path)
    results = []
    for page_no in range(len(doc)):
        page = doc[page_no]
        page_text = page.get_text()
        page_images = []
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_bytes = pix.tobytes("png")
            image_path = os.path.join(out_dir, f"page_{page_no+1}_img_{img_index+1}.png")
            with open(image_path, "wb") as f:
                f.write(img_bytes)
            try:
                ocr_text = pytesseract.image_to_string(Image.open(image_path))
            except Exception:
                print(f"OCR failed for {image_path}")
                ocr_text = ""
            page_images.append({"image_path": image_path, "ocr_text": ocr_text})
            pix = None
        results.append({
            "page": page_no+1,
            "text": page_text,
            "images": page_images
        })
    
    print(results)
    return results




# pdf_path = r"C:\Users\GathalaDilipKumar\Mygit\AIAgents_AgenticAI_AgenticRAG_MCP_A2A\AgenticRAGs\AmpdEnergy_doc_chatbot\Ampd Enertainer User Manual (NCM) - Rev 2.3.pdf"

# extract_text_and_images(pdf_path, out_dir="extracted_images")