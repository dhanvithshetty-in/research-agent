import os

# Parsing layer: strip document formatting down to raw text
# Supported: PDF (pdfplumber + OCR fallback), DOCX (python-docx), plain TXT
# OCR uses pytesseract -- requires Tesseract system binary (https://github.com/UB-Mannheim/tesseract/wiki)


def load_document_layer(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return _extract_pdf_text(filepath)
    elif ext == ".docx":
        return _extract_docx_text(filepath)
    elif ext == ".txt":
        return _extract_txt_text(filepath)
    else:
        raise ValueError(f"unsupported format: {ext}")


def _extract_pdf_text(filepath):
    import pdfplumber
    buffer = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                buffer.append(content)
    text = "\n".join(buffer)
    if text.strip():
        return text
    return _ocr_pdf(filepath)


def _ocr_pdf(filepath):
    try:
        import pdf2image
        import pytesseract
    except ImportError:
        return ""
    try:
        images = pdf2image.convert_from_path(filepath, dpi=300)
    except Exception:
        return ""
    buffer = []
    for img in images:
        content = pytesseract.image_to_string(img, lang="eng")
        if content.strip():
            buffer.append(content)
    return "\n".join(buffer)


def _extract_docx_text(filepath):
    from docx import Document
    doc = Document(filepath)
    buffer = []
    for para in doc.paragraphs:
        if para.text.strip():
            buffer.append(para.text)
    return "\n".join(buffer)


def _extract_txt_text(filepath):
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def discover_candidates(resume_dir):
    # File buffer protection: skip hidden OS artifacts (.DS_Store, thumbs.db)
    # and feed only active candidate payloads into the assessment pipeline
    valid = []
    for fname in os.listdir(resume_dir):
        if fname.startswith("."):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext in (".pdf", ".docx", ".txt"):
            valid.append(os.path.join(resume_dir, fname))
    valid.sort()
    return valid
