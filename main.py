import os
import pandas as pd
import psycopg2
import re
from PIL import Image
from fuzzywuzzy import fuzz
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import uvicorn
from dotenv import load_dotenv
from sqlalchemy import create_engine
import google.generativeai as genai
import json
import nest_asyncio

# Memuat variabel lingkungan dari file .env
load_dotenv()

# Mengambil variabel lingkungan. Menggunakan DB_DB untuk konsistensi.
DB_NAME = os.getenv("DB_DB")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

class KeywordNormalizer:
    """Class untuk mengelola dan melakukan normalisasi keywords dari database."""
    def __init__(self):
        self.keywords_map = {}
        self.load_keywords()

    def load_keywords(self):
        """Memuat keyword normalisasi dari database."""
        self.keywords_map.clear()
        try:
            with psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            ) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT keyword, normalized_value FROM keywords")
                    rows = cursor.fetchall()
                    for keyword, normalized_value in rows:
                        self.keywords_map[keyword.lower()] = normalized_value.lower()
            print("Keywords berhasil dimuat dari database.")
        except Exception as e:
            print(f"Error loading keywords from DB: {e}")

    def normalize_text(self, text: str) -> tuple[str, dict]:
        """Normalisasi teks dan identifikasi typo."""
        original_words = re.findall(r'\b\w+\b', text.lower())
        normalized_words = []
        typos = {}

        for word in original_words:
            if word in self.keywords_map:
                normalized_value = self.keywords_map[word]
                typos[word] = normalized_value
                normalized_words.append(normalized_value)
            else:
                normalized_words.append(word)

        normalized_text = " ".join(normalized_words)
        return normalized_text, typos

    def add_keyword_to_db(self, keyword: str, normalized_value: str):
        """Menambahkan keyword baru ke database lalu update map di memori."""
        try:
            with psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            ) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO keywords (keyword, normalized_value)
                        VALUES (%s, %s)
                        ON CONFLICT (keyword) DO UPDATE SET normalized_value = EXCLUDED.normalized_value
                    """, (keyword.lower(), normalized_value.lower()))
                    conn.commit()
            self.keywords_map[keyword.lower()] = normalized_value.lower()
            print(f"Keyword '{keyword}' berhasil ditambahkan/diperbarui di database.")
        except Exception as e:
            print(f"Error adding keyword to DB: {e}")

def load_data_pusat():
    """Memuat data dari tabel data_pusat."""
    try:
        engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        df = pd.read_sql_query("SELECT id, description, brand FROM public.data_pusat", con=engine)
        df = df.dropna()
        descriptions = df["description"].astype(str).str.lower().tolist()
        data_list = df.to_dict('records')
        print("Data pusat berhasil dimuat.")
        return df, descriptions, data_list
    except Exception as e:
        print(f"Error loading data from DB: {e}")
        return pd.DataFrame(), [], []

# Konfigurasi Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel("gemini-2.5-flash")

def OCR_Gemini(image_path: str) -> dict:
    """Melakukan OCR pada gambar struk menggunakan Gemini API."""
    try:
        img = Image.open(image_path)
        prompt = """
        Ini adalah struk belanja. Tolong ekstrak informasinya dalam format JSON dengan field:
        - invoice_number (string)
        - phone (string)
        - alamat (string)
        - email (string)
        - nama_toko (string)
        - tanggal (string, format DD/MM/YYYY)
        - daftar_barang (array of objects: nama, qty, harga_satuan, subtotal)
        - total_belanja (number)
        Jika ada informasi yang tidak jelas, isi dengan null.
        Hanya kembalikan JSON-nya saja, tanpa penjelasan atau markdown formatting.
        """
        response = model_gemini.generate_content([prompt, img])
        text = response.text
        cleaned = re.sub(r'^```json|```$', '', text, flags=re.MULTILINE).strip()
        return json.loads(cleaned)
    except Exception as e:
        print(f"Error during Gemini OCR: {e}")
        return {"daftar_barang": []}

def match_items(items: list, descriptions: list, data_list: list, normalizer: KeywordNormalizer, threshold: int = 58) -> list:
    """Mencocokkan item OCR dengan data pusat."""
    result = []
    
    for item in items:
        ocr_item_name = item.get('nama', '').lower()
        normalized_name, typos = normalizer.normalize_text(ocr_item_name)
        typo_dict = {f"additionalProp{i+1}": val for i, (key, val) in enumerate(typos.items())}

        best_score = 0
        best_match_data = None
        
        for i, desc in enumerate(descriptions):
            score = fuzz.token_set_ratio(normalized_name, desc)
            if score > best_score:
                best_score = score
                best_match_data = data_list[i]

        # Logika pencocokan yang diperbarui
        # Hanya jika nama OCR mengandung "nivea" DAN skor pencocokan lebih dari threshold
        # maka item dianggap "benar"
        is_nivea = "nivea" in normalized_name
        is_match_found = best_score >= threshold and is_nivea
        
        # Jika bukan Nivea, hasil selalu "salah"
        if not is_nivea:
            is_match_found = False

        # Mengembalikan data dengan struktur baru
        item_result = {
            "id": int(best_match_data.get("id")) if is_match_found and best_match_data else 0,
            "name": best_match_data.get("description") if is_match_found and best_match_data else "",
            "data": best_match_data.get("brand") if is_match_found and best_match_data else "",
            "ocr_item": {
                "name": item.get('nama', ''),
                "quantity": float(item.get('qty', 0) or 0),
                "price": float(item.get('harga_satuan', 0) or 0),
                "total": float(item.get('subtotal', 0) or 0),
            },
            "ocr_result": {
                "accuration": round(best_score / 100, 4) if is_match_found else 0.0,
                "typo": typo_dict,
                "normalisasi": normalized_name,
                "hasil": "benar" if is_match_found else "salah"
            }
        }
        result.append(item_result)
    return result

# --- Pydantic Models untuk FastAPI ---
class OCRItem(BaseModel):
    name: Optional[str]
    quantity: Optional[float]
    price: Optional[float]
    total: Optional[float]

class MatchResult(BaseModel):
    accuration: Optional[float]
    typo: Optional[Dict[str, str]]
    normalisasi: Optional[str]
    hasil: Optional[str]

class ItemMatched(BaseModel):
    id: Optional[int]
    name: Optional[str]
    data: Optional[str]
    ocr_item: OCRItem
    ocr_result: MatchResult

class Merchant(BaseModel):
    name: Optional[str]
    address: Optional[str]
    phone: Optional[str]
    email: Optional[str]

class FinalOutput(BaseModel):
    invoice_number: Optional[str]
    tanggal: Optional[str]
    merchant: Optional[Merchant]
    items: List[ItemMatched]
    grand_total: Optional[float]

class NormalizationRequest(BaseModel):
    keyword: str
    normalized_value: str

# --- Inisialisasi Aplikasi FastAPI ---
app = FastAPI()
df, descriptions, data_list = load_data_pusat()
normalizer = KeywordNormalizer()

@app.post("/struk-batch", response_model=List[FinalOutput])
async def struk_batch(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            path = tmp.name

        try:
            OCRData = OCR_Gemini(path)
            items = OCRData.get('daftar_barang', [])
            matched = match_items(items, descriptions, data_list, normalizer)

            result = {
                "invoice_number": OCRData.get('invoice_number'),
                "tanggal": OCRData.get('tanggal'),
                "merchant": {
                    "name": OCRData.get('nama_toko'),
                    "address": OCRData.get('alamat'),
                    "phone": OCRData.get('phone'),
                    "email": OCRData.get('email')
                },
                "items": matched,
                "grand_total": float(OCRData.get('total_belanja', 0)) if OCRData.get('total_belanja') is not None else 0
            }
            results.append(result)
        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")
            results.append({
                "error": f"Failed to process file {file.filename}.",
                "detail": str(e)
            })
        finally:
            try:
                os.remove(path)
            except Exception as e:
                print(f"Gagal menghapus file {path}: {e}")
    return results

@app.post("/add-normalization")
def add_normalization(data: NormalizationRequest):
    """Endpoint untuk menambahkan keyword normalisasi baru ke DB"""
    normalizer.add_keyword_to_db(data.keyword, data.normalized_value)
    return {
        "message": f"Normalisasi '{data.keyword}' -> '{data.normalized_value}' berhasil disimpan ke DB"
    }

@app.get("/")
def health_check():
    return {"status": "running"}

if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

