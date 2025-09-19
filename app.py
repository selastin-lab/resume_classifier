# app.py (FULL - manual roles assignment)
import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import random

from resume_parser import (
    extract_text_bytes,
    extract_skills,
    extract_experience_years,
    extract_education,
    cosine_similarity_np,
)

# -------- Config & model paths ----------
MODEL_PATH = "model_files/bert_clf.joblib"
st.set_page_config(page_title="Resume Classifier — Recruiter Toolkit", layout="wide")

# --------- Load saved classifier ----------
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. Run `python train.py` first.")
    st.stop()

model_obj = joblib.load(MODEL_PATH)
clf = model_obj["clf"] if isinstance(model_obj, dict) and "clf" in model_obj else model_obj

# --------- Load BERT tokenizer/model (cached) ---------------------------
@st.cache_resource
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

tokenizer, bert_model, device = load_bert()

# --------- Embedding helper ----------
def compute_embeddings(texts, tokenizer, model, device="cpu", batch_size=32, progress_fn=None):
    model.eval()
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * mask_expanded, 1)
            counts = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled = (summed / counts).cpu().numpy()
            all_embs.append(pooled)
            if progress_fn:
                progress_fn(min((i + batch_size) / max(1, len(texts)), 1.0))
    if len(all_embs) == 0:
        hidden_size = getattr(model.config, "hidden_size", 768)
        return np.zeros((0, hidden_size))
    return np.vstack(all_embs)

# --------- UI layout ----------
st.title("selastin 📂 Resume Classifier — Recruiter Toolkit")
st.write("Upload multiple resumes (PDF/DOCX/TXT). Features: confidence score, skill extraction, experience, dashboard, search/filter, job-role matching, bulk progress and export.")

col1, col2 = st.columns([2, 1])

with col2:
    st.header("Actions")
    save_separate = st.checkbox("Save into folders by label", value=True)
    batch_size = st.number_input("Embedding batch size", min_value=4, max_value=128, value=32, step=4)
    top_k = st.number_input("Top matches for Job Description", min_value=1, max_value=20, value=5, step=1)

# file uploader
uploaded_files = st.file_uploader("Drop resumes here (PDF / DOCX / TXT)", accept_multiple_files=True, type=["pdf", "docx", "txt"])

# job description panel
st.subheader("Job Description → Match Resumes")
job_desc = st.text_area("Paste a job description (optional) to find best candidates", height=120)
match_btn = st.button("Find Matches")

# Manual roles for demonstration
manual_roles = ["Data Scientist", "Doctor", "Teacher", "AI Engineer", "Designer", "Developer"]

# Main processing
if uploaded_files:
    filenames = [f.name for f in uploaded_files]
    raw_bytes = [f.read() for f in uploaded_files]

    # Extract text
    st.info("Extracting text from uploaded resumes...")
    texts = [extract_text_bytes(b, n) for b, n in zip(raw_bytes, filenames)]
    empty_mask = [not bool(t.strip()) for t in texts]
    texts_for_emb = [t if t.strip() else " " for t in texts]

    st.info(f"Computing embeddings for {len(texts)} files (batch {batch_size})...")
    pb = st.progress(0.0)
    def progress_fn(p): pb.progress(float(p))
    embeddings = compute_embeddings(texts_for_emb, tokenizer, bert_model, device=device, batch_size=batch_size, progress_fn=progress_fn)

    # Predict with probabilities (optional, can ignore if manual roles)
    try:
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(embeddings)
            confidences = proba.max(axis=1)
            preds = clf.classes_[proba.argmax(axis=1)]
        else:
            preds = clf.predict(embeddings)
            confidences = np.ones(len(preds)) * 0.6
    except Exception:
        confidences = np.ones(len(filenames)) * 0.6

    # ---- Assign manual roles ----
    final_labels = [
        "Unknown" if empty else manual_roles[i % len(manual_roles)]
        for i, empty in enumerate(empty_mask)
    ]

    # Extract skills/experience/education
    skills_list = []
    years_exp_list = []
    education_list = []
    for t in texts:
        skills_list.append(", ".join(extract_skills(t)))
        years_exp_list.append(extract_experience_years(t) or "")
        education_list.append(", ".join(extract_education(t)))

    # Build DataFrame
    df_rows = []
    for i, name in enumerate(filenames):
        df_rows.append({
            "filename": name,
            "label": final_labels[i],
            "confidence": float(confidences[i]) if confidences is not None else None,
            "skills": skills_list[i],
            "experience_years": years_exp_list[i],
            "education": education_list[i],
            "path": "",
        })
    df = pd.DataFrame(df_rows)

    # Save files into category folders
    if save_separate:
        for i, (name, label, raw) in enumerate(zip(filenames, final_labels, raw_bytes)):
            dest_dir = os.path.join("classified_resumes", label)
            os.makedirs(dest_dir, exist_ok=True)
            save_path = os.path.join(dest_dir, name)
            with open(save_path, "wb") as out_f:
                out_f.write(raw)
            df.at[i, "path"] = save_path

    # Left: Results & Filters, Right: Dashboard
    left, right = st.columns([3, 2])

    # --- Left: Filters & Results ---
    with left:
        st.subheader("Results — Search & Filter")
        roles = ["All"] + sorted([str(x) for x in df["label"].unique()])
        q_name = st.text_input("Search by filename or candidate name")
        q_skill = st.text_input("Search by skill (e.g., python, excel)")
        q_role = st.selectbox("Filter by role", options=roles, index=0)

        filtered = df.copy()
        if q_name:
            filtered = filtered[filtered["filename"].str.contains(q_name, case=False, na=False)]
        if q_skill:
            filtered = filtered[filtered["skills"].str.contains(q_skill, case=False, na=False)]
        if q_role != "All":
            filtered = filtered[filtered["label"] == q_role]

        def fmt_conf(v):
            try:
                return f"{v*100:.0f}%"
            except:
                return ""
        filtered_display = filtered.copy()
        filtered_display["confidence"] = filtered_display["confidence"].apply(fmt_conf)

        st.dataframe(filtered_display[["filename", "label", "confidence", "skills", "experience_years", "education"]])

        # CSV / Excel export
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Export CSV of results", data=csv_bytes, file_name="resume_results.csv", mime="text/csv")

        try:
            import openpyxl
            out = io.BytesIO()
            filtered.to_excel(out, index=False, engine="openpyxl")
            out.seek(0)
            st.download_button("Export Excel", data=out, file_name="resume_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            pass

    # --- Right: Dashboard ---
    with right:
        st.subheader("Dashboard")
        counts = df["label"].value_counts()
        st.write("Role distribution")
        st.bar_chart(counts)

        st.write("Top skills (by occurrences)")
        all_sk = []
        for s in df["skills"].str.split(","):
            if isinstance(s, list):
                for x in s:
                    x = x.strip()
                    if x:
                        all_sk.append(x)
        sk_series = pd.Series(all_sk).value_counts().head(20)
        st.table(sk_series)

    # --- Job description matching ---
    if job_desc and match_btn:
        st.subheader("Job → Candidate Matching")
        jd_text = job_desc.strip()
        if jd_text:
            jd_emb = compute_embeddings([jd_text], tokenizer, bert_model, device=device, batch_size=1)
            sims = cosine_similarity_np(jd_emb, embeddings).flatten()
            top_idx = np.argsort(-sims)[:top_k]
            match_rows = []
            for idx in top_idx:
                score = float(sims[idx])
                match_rows.append({
                    "filename": filenames[idx],
                    "predicted_label": final_labels[idx],
                    "confidence": float(confidences[idx]) if confidences is not None else None,
                    "similarity": score,
                    "skills": skills_list[idx],
                    "experience_years": years_exp_list[idx],
                })
            match_df = pd.DataFrame(match_rows)
            match_df["similarity"] = match_df["similarity"].apply(lambda x: f"{x*100:.0f}%")
            match_df["confidence"] = match_df["confidence"].apply(lambda x: f"{x*100:.0f}%" if pd.notnull(x) else "")
            st.table(match_df)

    st.success("Classification complete.")

else:
    st.info("Upload one or more resumes (PDF/DOCX/TXT). The app supports bulk uploads (100+).")
