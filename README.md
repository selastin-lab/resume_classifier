Resume Classifier
Overview

The Resume Classifier is a machine learning-based application that automatically classifies resumes into different job categories. This project uses natural language processing (NLP) techniques and a transformer-based model (like BERT) to understand and categorize resumes efficiently.

Features

Resume Parsing: Extracts textual content from resumes.

Job Category Classification: Classifies resumes into predefined job roles such as Data Scientist, Software Engineer, Analyst, etc.

High Accuracy: Uses a pre-trained BERT model fine-tuned on resume data for robust predictions.

Easy to Use: Simple interface for uploading resumes and getting predictions.

Technologies Used

Programming Language: Python

Libraries & Frameworks:

Transformers (Hugging Face)

PyTorch

scikit-learn

pandas

numpy

Flask / Streamlit (optional for web interface)

Project Structure
resume_classifier/
│
├── data/                   # Dataset folder
│   ├── train.csv
│   └── test.csv
│
├── models/                 # Saved trained models
│   └── bert_resume_model.pt
│
├── scripts/                # Python scripts
│   ├── preprocess.py       # Data preprocessing
│   ├── train.py            # Training script
│   └── predict.py          # Resume prediction script
│
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

Installation

Clone the repository:

git clone https://github.com/your-username/resume-classifier.git
cd resume-classifier


Create a virtual environment and activate it:

python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows


Install dependencies:

pip install -r requirements.txt

Usage

Train the Model

python scripts/train.py


Predict a Resume Category

python scripts/predict.py --resume_path path/to/resume.pdf


Optional: Run a web interface (if implemented with Streamlit or Flask)

streamlit run app.py

Dataset

The dataset should contain resumes along with corresponding job category labels.

Example CSV format:

Resume_Text,Category
"Experienced Python developer with 3 years...", "Software Engineer"
"Data analyst with strong SQL and Tableau skills...", "Data Analyst"

Contributing

Contributions are welcome! Feel free to:

Improve model performance

Add more job categories

Implement better resume parsing

Add a user-friendly web interface

License

This project is licensed under the MIT License.
