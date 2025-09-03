# 🖼️ Image Caption Generator with CNN & LSTM

## 📌 Overview
This project implements an **Image Caption Generator** that can automatically describe images in natural language. It combines **Computer Vision** and **Natural Language Processing** using a hybrid **CNN–RNN** deep learning architecture.  

We use a **Convolutional Neural Network (CNN)** (VGG16 / Xception) to extract features from images and a **Long Short-Term Memory (LSTM)** network to generate descriptive captions.

---

## 🚀 Project Workflow
1. **Image Feature Extraction**  
   - Pretrained **VGG16** is used to extract image embeddings.  
   - Features are stored in a `.pkl` file for efficiency.  

2. **Caption Preprocessing**  
   - Captions are cleaned, lowercased, and tokenized.  
   - Start (`startseq`) and end (`endseq`) tokens are added.  

3. **Data Preparation**  
   - Captions are converted into sequences.  
   - Input sequences are padded to a fixed length.  
   - Data is split into training (85%) and testing (15%).  

4. **Model Architecture**  
   - CNN (VGG16/Xception) extracts image features.  
   - LSTM generates sequential text.  
   - Attention mechanism improves alignment between image features and words.  

5. **Training**  
   - Loss: `categorical_crossentropy`  
   - Optimizer: `Adam`  
   - EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint used for stable training.  

6. **Evaluation**  
   - Model performance is measured using **BLEU Score**.  
   - BLEU-1: `0.53`, BLEU-2: `0.30`  

7. **Prediction & Visualization**  
   - Given an image, the model generates captions.  
   - Actual vs Predicted captions are compared.

---

## 📂 Dataset
- **Flickr8k Dataset** (8,091 images with 5 captions each)  
- Download: [Kaggle – Flickr8k](https://www.kaggle.com/datasets)  
- Larger datasets like **Flickr30k** and **MSCOCO** can also be used for better accuracy.  

---

## 📦 Requirements
- Python 3.7+
- TensorFlow / Keras
- NumPy, Pandas
- NLTK
- tqdm
- Matplotlib, PIL

Install dependencies:
```bash
pip install tensorflow keras numpy pandas nltk tqdm matplotlib pillow
