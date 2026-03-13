# AI Text Detection using Machine Learning and BERT

This project detects AI-generated text using multiple machine learning techniques including stylometric analysis, TF-IDF models, and a fine-tuned BERT deep learning model.

---

## Project Overview

With the rapid growth of generative AI tools, distinguishing between human-written and AI-generated content has become important in academic and professional environments. This project builds a machine learning pipeline capable of identifying AI-generated text.

The system analyzes writing style, vocabulary usage, sentence structure, and contextual patterns to determine whether a document is written by a human or an AI model.

---

## Features

- Stylometric feature extraction
- TF-IDF based baseline classifier
- Combined machine learning model
- Deep learning detection using BERT
- Paragraph-level AI probability detection
- Batch document analysis
- AI vs Human percentage estimation

---

## Technologies Used

- Python
- PyTorch
- HuggingFace Transformers
- Scikit-learn
- Pandas
- NumPy

---

## Project Structure
![alt text](<Screenshot 2026-03-13 at 3.27.29 PM.png>)


---

## Dataset

Human written essays were obtained from the **ASAP-AES dataset**.

AI-generated essays were created using multiple large language models to avoid bias and improve generalization.

---

## Model Pipeline

1. Dataset collection
2. Text preprocessing
3. Stylometric feature extraction
4. TF-IDF vectorization
5. Machine learning classification
6. BERT deep learning model training
7. AI probability prediction

---

## Example Output

---

## Dataset

Human written essays were obtained from the **ASAP-AES dataset**.

AI-generated essays were created using multiple large language models to avoid bias and improve generalization.

---

## Model Pipeline

1. Dataset collection
2. Text preprocessing
3. Stylometric feature extraction
4. TF-IDF vectorization
5. Machine learning classification
6. BERT deep learning model training
7. AI probability prediction

---

## Example Output
Paragraph 1: Human (AI probability: 0.10)
Paragraph 2: Human (AI probability: 0.06)
Paragraph 3: AI (AI probability: 0.84)

Final Result:
AI Used: 33.47%
Human Written: 66.53%


---

## Future Improvements

- OCR support for scanned PDFs
- Web interface for real-time AI detection
- Model optimization for faster inference
- Larger dataset for improved accuracy

---

## Author
Gazal Yadav  
CSE (AI/ML) Student