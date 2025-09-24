# create_index.py
import argparse, json, os
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from utils.pdf_utils import extract_text_and_images
from tqdm import tqdm


def build_index(pdf_path, index_path="index.faiss", meta_path="meta.json", image_dir="extracted_images"):

    # 1) Extract text and images together
    print("Extracting text and images from PDF...")
    results = extract_text_and_images(pdf_path, out_dir=image_dir)

    # 2) Chunk text and images
    print("Chunking text and images...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=120, length_function=len
    )


    chunks = []
    for page_obj in results:
        page_num = page_obj["page"]
        page_text = page_obj["text"]
        image_paths = []
        ocr_texts = []
        for img in page_obj["images"]:
            image_paths.append(img["image_path"])
            if img["ocr_text"].strip():
                ocr_texts.append(img["ocr_text"].strip())
            else:
                ocr_texts.append(f"[Image present on page {page_num}]")

        # Merge OCR texts into the main text
        merged_text = page_text.strip()
        if ocr_texts:
            merged_text += "\n" + "\n".join([f"[Image OCR] {t}" if not t.startswith("[Image present") else t for t in ocr_texts])

        # Split merged text into chunks
        if merged_text:
            doc = Document(page_content=merged_text, metadata={"page": page_num})
            sub_docs = splitter.split_documents([doc])
            for sd in sub_docs:
                chunks.append({
                    "page": page_num,
                    "text": sd.page_content,
                    "image_paths": image_paths if image_paths else None
                })

    print(f"Total chunks (text + images): {len(chunks)}")

    # 4) Embeddings using SentenceTransformer
    print("Loading embedding model and computing embeddings...")
    embed_model_name = "all-MiniLM-L6-v2"
    embedder = SentenceTransformer(embed_model_name)
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # 5) Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product with normalized vectors -> cosine
    # Normalize vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normed = embeddings / (norms + 1e-12)
    index.add(embeddings_normed.astype('float32'))

    # Save faiss index
    print(f"Saving FAISS index to {index_path}...")
    faiss.write_index(index, index_path)


    # Save metadata (chunks + image_paths)
    print(f"Saving metadata to {meta_path}...")
    meta_objs = []
    for i, c in enumerate(chunks):
        meta = {
            "id": i,
            "page": c.get("page"),
            "text": c.get("text"),
            "image_paths": c.get("image_paths", None)
        }
        meta_objs.append(meta)
    with open(meta_path, "w", encoding="utf8") as f:
        json.dump(meta_objs, f, ensure_ascii=False, indent=2)

    print("✅ Index build complete. Chunks:", len(chunks))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", required=True, help="Path to AmpD Enertainer User Manual PDF")
    parser.add_argument("--index", default="index.faiss")
    parser.add_argument("--meta", default="meta.json")
    args = parser.parse_args()
    build_index(args.doc, args.index, args.meta)
