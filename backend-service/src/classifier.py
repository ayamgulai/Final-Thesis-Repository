import joblib
import torch
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bs4 import BeautifulSoup
import html
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class TicketClassifier:
    def __init__(self, svm_path, indobert_dir):
        # 1. Load SVM Pipeline
        self.svm_pipeline = joblib.load(svm_path)
        
        # 2. Load Label Encoder
        self.label_encoder = joblib.load(f"{indobert_dir}/label_encoder.joblib")

        # 3. Initialize Sastrawi Stemmer (dibuat sekali agar tidak ada overhead
        #    konstruksi factory ~100ms pada setiap pemanggilan predict_svm)
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
            
        # 4. Load IndoBERT
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(indobert_dir)
        self.indobert_model = AutoModelForSequenceClassification.from_pretrained(
            indobert_dir, 
            use_safetensors=True
        ).to(self.device)
        self.indobert_model.eval()

    def preprocess_for_svm(self, text):
        """Replika dari fungsi preprocess() di notebook exploratory
        yang menghasilkan kolom 'clean_text' untuk training SVM.
        
        Pipeline ini harus identik dengan preprocessing saat training:
        HTML decode → strip tag → lowercase → hapus punctuation →
        normalisasi whitespace → stemming (Sastrawi).
        """
        if not isinstance(text, str): return ""
        # 1. Decode entitas HTML
        text = html.unescape(text)
        # 2. Hapus tag HTML
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
        # 3. Lowercase
        text = text.lower()
        # 4. Hapus punctuation (tanda baca), termasuk underscore
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        # 5. Normalisasi whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # 6. Stemming — mereduksi kata berimbuhan ke bentuk dasar
        #    agar sesuai dengan vocabulary TF-IDF yang dibangun saat training
        text = self.stemmer.stem(text)
        return text

    def preprocess_for_bert(self, text):
        if not isinstance(text, str): return ""
        text = html.unescape(text)
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
        text = text.lower().strip()
        return text # Stopwords & tanda baca TETAP DIPERTAHANKAN

    def predict_svm(self, text):
        # Preprocess teks agar sesuai dengan clean_text yang dipakai saat training
        text = self.preprocess_for_svm(text)
        
        # Gunakan predict_proba untuk mendapatkan probabilitas setiap kelas
        probabilities = self.svm_pipeline.predict_proba([text])[0]
        
        # Ambil indeks kelas dengan probabilitas tertinggi
        best_class_idx = np.argmax(probabilities)
        confidence = probabilities[best_class_idx]
        
        # Ambil label kelas aslinya menggunakan atribut .classes_ dari pipeline
        pred_label = self.svm_pipeline.classes_[best_class_idx]
        
        # Cek apakah elemen array sudah berupa string (label langsung)
        # Ini terjadi ketika pipeline sudah di-train dengan target string
        if isinstance(pred_label, (str, np.str_)):
            return str(pred_label), float(confidence)
        
        # Jika hasil berupa angka (integer encoding), gunakan inverse_transform
        decoded_labels = self.label_encoder.inverse_transform([pred_label])
        return str(decoded_labels[0]), float(confidence)

    def predict_indobert(self, text):
        text = self.preprocess_for_bert(text)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.indobert_model(**inputs)
            logits = outputs.logits
            
            probs = torch.softmax(logits, dim=-1)
            pred_id = torch.argmax(probs, dim=1).item() # Menghasilkan integer tunggal
            confidence = probs[0, pred_id].item()
        # Karena pred_id adalah integer tunggal, bungkus dalam list [pred_id]
        # Kemudian ambil elemen  untuk mendapatkan string murni (misal: "BK")
        pred_label = self.label_encoder.inverse_transform([pred_id])[0]
        return str(pred_label), confidence