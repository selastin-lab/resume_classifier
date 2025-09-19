# resume_parser.py
import re
import io
import os
from tempfile import NamedTemporaryFile

import numpy as np

# Optional imports (OCR)
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import docx
except Exception:
    docx = None

try:
    import fitz  # PyMuPDF
    from PIL import Image
    import pytesseract
except Exception:
    fitz = None
    Image = None
    pytesseract = None

# A starter skills list (extend this list with domain-specific skills)
COMMON_SKILLS = [
    "python","java","c++","c#","sql","excel","pandas","numpy","scikit-learn","tensorflow","keras",
    "pytorch","aws","azure","gcp","docker","kubernetes","react","node.js","javascript","html","css",
    "tableau","powerbi","spark","hadoop","r","matlab","rest api","api","ml","nlp",
    "computer vision","opencv","photoshop","illustrator","autocad","solidworks","agile","scrum"
]
COMMON_SKILLS = [s.lower() for s in COMMON_SKILLS]

EDUCATION_KEYWORDS = ["bachelor", "b.sc", "b.tech", "b.e", "master", "m.sc", "m.tech", "m.e", "mba", "phd", "doctor", "bcom", "msc", "bs", "ms"]

Y_EXP_REGEX = [
    r'(\d{1,2})\s*\+\s*years',          # "5+ years"
    r'(\d{1,2})\s+years',               # "5 years"
    r'(\d{1,2})\s*-?\s*(\d{1,2})\s+years' # "3-5 years"
]

def extract_text_bytes(file_bytes: bytes, filename: str) -> str:
    """
    Extract text from bytes. Tries PyPDF2 and python-docx. If PDF text is empty and
    PyMuPDF + pytesseract are available, performs OCR on each page image.
    """
    ext = os.path.splitext(filename)[1].lower()
    text = ""

    # write to temp file
    with NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if ext == ".txt":
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        elif ext == ".pdf" and PyPDF2:
            try:
                with open(tmp_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for p in reader.pages:
                        text += p.extract_text() or ""
            except Exception:
                text = ""

            # OCR fallback for image-PDFs
            if not text.strip() and fitz and pytesseract and Image:
                pdf = fitz.open(tmp_path)
                for page in pdf:
                    pix = page.get_pixmap(dpi=200)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img)
                    text += "\n" + ocr_text

        elif ext == ".docx" and docx:
            try:
                d = docx.Document(tmp_path)
                text = "\n".join(p.text for p in d.paragraphs)
            except Exception:
                text = ""
        else:
            # fallback
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return text or ""

def extract_skills(text: str, skills_list=None):
    text_l = text.lower()
    skills_list = skills_list or COMMON_SKILLS
    found = set()
    for s in skills_list:
        if s in text_l:
            found.add(s)
    # also simple token approach for camel-case tokens (e.g., "PowerBI")
    tokens = re.findall(r'[A-Za-z+#\.\-]{2,}', text)
    for t in tokens:
        tl = t.lower()
        if tl in skills_list:
            found.add(tl)
    return sorted(found)

def extract_experience_years(text: str):
    text_l = text.lower()
    years = []
    for regex in Y_EXP_REGEX:
        for m in re.finditer(regex, text_l):
            if m:
                groups = m.groups()
                if len(groups) == 2 and groups[1] and groups[0]:
                    try:
                        low = int(groups[0])
                        high = int(groups[1])
                        years.append((low+high)//2)
                    except:
                        pass
                else:
                    try:
                        years.append(int(groups[0]))
                    except:
                        pass
    if years:
        return max(years)
    # fallback: search for "experience: X years"
    m = re.search(r'experience[:\s]+(\d{1,2})', text_l)
    if m:
        try:
            return int(m.group(1))
        except:
            pass
    return None

def extract_education(text: str):
    text_l = text.lower()
    found = []
    for kw in EDUCATION_KEYWORDS:
        if kw in text_l:
            found.append(kw)
    return sorted(set(found))

# Cosine similarity helper
def cosine_similarity_np(A: np.ndarray, B: np.ndarray):
    # A: (n, d), B: (m, d) -> returns (n, m)
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return np.dot(A_norm, B_norm.T)
