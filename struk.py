import os
import pandas as pd
import psycopg2
import re
import json
import logging
from PIL import Image
from fuzzywuzzy import fuzz
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import uvicorn
from dotenv import load_dotenv
from sqlalchemy import create_engine
import google.generativeai as genai
import nest_asyncio
from functools import lru_cache
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import asyncio
import io # Import library io untuk in-memory file handling

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Get environment variables, using default values to prevent crashes
DB_NAME = os.getenv("DB_DB")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
OCR_CONCURRENCY = int(os.getenv("OCR_CONCURRENCY", "8"))

class KeywordNormalizer:
    """Class to manage and perform keyword normalization from a database."""
    def __init__(self):
        self.keywords_map = {}
        self.load_keywords()
        self.learn_from_file('normalisasi.txt')

    def _get_db_conn(self):
        """Helper to create a database connection."""
        return psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )

    def load_keywords(self):
        """Loads normalization keywords from the database."""
        self.keywords_map.clear()
        try:
            with self._get_db_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT keyword, normalized_value FROM keywords")
                    rows = cursor.fetchall()
                    for keyword, normalized_value in rows:
                        self.keywords_map[keyword.lower()] = normalized_value.lower()
            logging.info("Keywords successfully loaded from the database.")
        except psycopg2.OperationalError as e:
            logging.error(f"Database connection error: {e}. Trying to create table 'keywords'...")
            self.create_keywords_table()
            self.load_keywords()
        except Exception as e:
            logging.error(f"Error loading keywords from DB: {e}")

    def learn_from_file(self, file_path: str):
        """Loads normalization rules from a text file and adds them to the database."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '->' in line:
                        keyword, normalized_value = line.split('->', 1)
                        self.add_keyword_to_db(keyword.strip(), normalized_value.strip())
            logging.info(f"Successfully loaded normalization rules from '{file_path}'.")
        except FileNotFoundError:
            logging.warning(f"Normalization file '{file_path}' not found. Skipping file import.")
        except Exception as e:
            logging.error(f"Error reading normalization file: {e}")

    def create_keywords_table(self):
        """Creates the 'keywords' table if it does not exist."""
        try:
            with self._get_db_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS keywords (
                            keyword VARCHAR(255) PRIMARY KEY,
                            normalized_value VARCHAR(255)
                        );
                    """)
                    conn.commit()
            logging.info("Table 'keywords' created successfully.")
        except Exception as e:
            logging.error(f"Failed to create table 'keywords': {e}")

    def normalize_text(self, text: str) -> tuple[str, dict]:
        """Normalizes text and identifies typos."""
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
        """Adds a new keyword to the database and updates the in-memory map."""
        try:
            with self._get_db_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO keywords (keyword, normalized_value)
                        VALUES (%s, %s)
                        ON CONFLICT (keyword) DO UPDATE SET normalized_value = EXCLUDED.normalized_value
                    """, (keyword.lower(), normalized_value.lower()))
                    conn.commit()
            self.keywords_map[keyword.lower()] = normalized_value.lower()
            logging.info(f"Keyword '{keyword}' successfully added/updated in the database.")
        except Exception as e:
            logging.error(f"Error adding keyword to DB: {e}")

    def add_keyword_if_typo(self, ocr_word: str, threshold: int = 90):
        """Checks if a word is a typo of an existing normalized word and adds it to the DB."""
        if ocr_word.lower() in self.keywords_map:
            return

        normalized_values = list(set(self.keywords_map.values()))
        best_match_value = None
        best_score = 0
        for value in normalized_values:
            score = fuzz.ratio(ocr_word.lower(), value.lower())
            if score >= threshold and score > best_score:
                best_score = score
                best_match_value = value
        
        if best_match_value:
            logging.info(f"Automatically learning typo: '{ocr_word}' -> '{best_match_value}' (Score: {best_score})")
            self.add_keyword_to_db(ocr_word, best_match_value)
            # Add the new typo to the in-memory map immediately
            self.keywords_map[ocr_word.lower()] = best_match_value.lower()


@lru_cache(maxsize=1)
def get_keyword_normalizer():
    """Singleton pattern for KeywordNormalizer."""
    return KeywordNormalizer()

@lru_cache(maxsize=1)
def load_data_pusat():
    """Loads data from the 'data_pusat' table and caches it."""
    try:
        engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        df = pd.read_sql_query("SELECT id, description, brand FROM public.data_pusat", con=engine)
        df = df.dropna()
        descriptions = df["description"].astype(str).str.lower().tolist()
        data_list = df.to_dict('records')
        logging.info("Central data loaded successfully.")
        return df, descriptions, data_list
    except Exception as e:
        logging.error(f"Error loading data from DB: {e}")
        return pd.DataFrame(), [], []

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel("gemini-1.5-flash")

def _process_ocr(image_bytes: bytes) -> dict: # Mengubah input menjadi bytes
    """Performs OCR on the receipt image using the Gemini API."""
    prompt = """
    This is a shopping receipt. Please extract the information in JSON format with the following fields:
    - invoice_number (string)
    - phone (string)
    - alamat (string)
    - email (string)
    - nama_toko (string)
    - tanggal (string, format DD/MM/YYYY)
    - daftar_barang (array of objects: nama, qty, harga_satuan, subtotal)
    - total_belanja (number)
    If any information is unclear, fill it with null.
    Return only the JSON, without any explanations or markdown formatting.
    """
    try:
        # Menggunakan io.BytesIO untuk membaca gambar dari memori
        img = Image.open(io.BytesIO(image_bytes))
        response = model_gemini.generate_content([prompt, img])
        text = response.text
        cleaned = re.sub(r'^```json|```$', '', text, flags=re.MULTILINE).strip()
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding failed from Gemini response: {e}")
        logging.debug(f"Raw Gemini response: {text}")
        raise ValueError("Failed to parse JSON from Gemini response.")
    except Exception as e:
        logging.error(f"Error during Gemini OCR: {e}")
        raise

def _match_items(items: list, descriptions: list, data_list: list, normalizer: KeywordNormalizer, threshold: int = 58) -> list:
    """Matches OCR items with central data."""
    allowed_brands = ["nivea", "biore", "posh", "khaf"]
    result = []
    
    for item in items:
        ocr_item_name = item.get('nama', '')
        normalized_name, typos = normalizer.normalize_text(ocr_item_name)
        
        # Peningkatan: Memproses setiap kata dari hasil OCR untuk mendeteksi typo
        ocr_words = re.findall(r'\b\w+\b', ocr_item_name.lower())
        for ocr_word in ocr_words:
            # Panggil fungsi auto-learning
            normalizer.add_keyword_if_typo(ocr_word)

        # Setelah learning, jalankan lagi normalisasi
        normalized_name, typos = normalizer.normalize_text(ocr_item_name)

        best_score = 0
        best_match_data = None
        
        for i, desc in enumerate(descriptions):
            score = fuzz.token_set_ratio(normalized_name, desc)
            if score > best_score:
                best_score = score
                best_match_data = data_list[i]

        is_brand_allowed = (
            best_match_data is not None and best_match_data.get("brand", "").lower() in allowed_brands
        )
        is_match_found = best_score >= threshold and is_brand_allowed

        try:
            item_result = {
                "id": int(best_match_data.get("id")) if is_match_found and best_match_data else None,
                "name": best_match_data.get("description") if is_match_found and best_match_data else None,
                "data": best_match_data.get("brand") if is_match_found and best_match_data else None,
                "ocr_result": {
                    "name": ocr_item_name,
                    "quantity": float(item.get('qty', 0) or 0),
                    "price": float(item.get('harga_satuan', 0) or 0),
                    "total": float(item.get('subtotal', 0) or 0),
                    "accuration": round(best_score / 100, 4) if is_match_found else 0.0,
                    "typo": typos,
                    "normalisasi": normalized_name,
                    "hasil": "benar" if is_match_found else "salah"
                }
            }
            result.append(item_result)
        except (ValueError, TypeError) as e:
            logging.error(f"Error converting data for item '{ocr_item_name}': {e}")
            result.append({
                "id": None,
                "name": None,
                "data": None,
                "ocr_result": {
                    "name": ocr_item_name,
                    "quantity": None,
                    "price": None,
                    "total": None,
                    "accuration": 0.0,
                    "typo": {},
                    "normalisasi": normalized_name,
                    "hasil": "error"
                }
            })
    return result

# Pydantic Models for FastAPI
class OCRResult(BaseModel):
    name: Optional[str] = None
    quantity: Optional[float] = None
    price: Optional[float] = None
    total: Optional[float] = None
    accuration: Optional[float] = None
    typo: Optional[Dict[str, str]] = Field(default_factory=dict)
    normalisasi: Optional[str] = None
    hasil: Optional[str] = None

class ItemMatched(BaseModel):
    id: Optional[int] = None
    name: Optional[str] = None
    data: Optional[str] = None
    ocr_result: OCRResult

class Merchant(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None

class FinalOutput(BaseModel):
    invoice_number: Optional[str] = None
    tanggal: Optional[str] = None
    merchant: Optional[Merchant] = None
    items: List[ItemMatched] = Field(default_factory=list)
    grand_total: Optional[float] = None
    error: Optional[str] = None
    detail: Optional[str] = None

class NormalizationRequest(BaseModel):
    keyword: str
    normalized_value: str

# FastAPI Application Initialization
app = FastAPI()

def save_results_to_db(output: FinalOutput):
    """Menyimpan hasil pemrosesan OCR ke database."""
    if output.error:
        logging.info("Melewatkan penyimpanan ke DB untuk proses yang gagal: %s", output.invoice_number or "Unknown")
        return

    conn = None
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        with conn.cursor() as cursor:
            tanggal_obj = None
            if output.tanggal:
                try:
                    # Coba parsing format DD/MM/YYYY
                    tanggal_obj = datetime.strptime(output.tanggal, '%d/%m/%Y').date()
                except (ValueError, TypeError):
                    # Coba format lain jika gagal, misal YYYY-MM-DD
                    try:
                        tanggal_obj = datetime.strptime(output.tanggal, '%Y-%m-%d').date()
                    except (ValueError, TypeError):
                        logging.warning("Tidak dapat mem-parsing tanggal '%s'. Menyimpan sebagai NULL.", output.tanggal)

            # Masukkan ke ocr_items
            cursor.execute("""
                INSERT INTO ocr_items (invoice_number, tanggal, nama_toko, alamat, phone, email, total_belanja)
                VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;
            """, (
                output.invoice_number,
                tanggal_obj,
                output.merchant.name if output.merchant else None,
                output.merchant.address if output.merchant else None,
                output.merchant.phone if output.merchant else None,
                output.merchant.email if output.merchant else None,
                output.grand_total
            ))
            receipt_id = cursor.fetchone()[0]

            # Masukkan ke ocr_results
            for item in output.items:
                ocr_res = item.ocr_result
                is_correct = True if ocr_res.hasil == 'benar' else (False if ocr_res.hasil == 'salah' else None)
                keywords_list = list(ocr_res.typo.keys()) if ocr_res.typo else []

                cursor.execute("""
                    INSERT INTO ocr_results (
                        receipt_id, ocr_name, ocr_quantity, ocr_price, ocr_total,
                        text_accuracy, matched_item_id, matched_name, keywords, is_correct
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """, (
                    receipt_id, ocr_res.name, ocr_res.quantity, ocr_res.price, ocr_res.total,
                    ocr_res.accuration, item.id, item.name, keywords_list, is_correct
                ))
            conn.commit()
            logging.info("Berhasil menyimpan hasil untuk invoice %s ke DB (receipt_id: %d)", output.invoice_number, receipt_id)

    except Exception as e:
        logging.error("Gagal menyimpan hasil untuk invoice %s ke DB: %s", output.invoice_number, e)
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def process_single_file(file_content: bytes, descriptions, data_list, normalizer):
    """Worker function for processing a single file using in-memory data."""
    try:
        ocr_data = _process_ocr(file_content) # Menggunakan bytes yang sudah dibaca
        items = ocr_data.get('daftar_barang', [])
        matched = _match_items(items, descriptions, data_list, normalizer)
        
        return FinalOutput(
            invoice_number=ocr_data.get('invoice_number'),
            tanggal=ocr_data.get('tanggal'),
            merchant=Merchant(
                name=ocr_data.get('nama_toko'),
                address=ocr_data.get('alamat'),
                phone=ocr_data.get('phone'),
                email=ocr_data.get('email')
            ),
            items=matched,
            grand_total=float(ocr_data.get('total_belanja', 0)) if ocr_data.get('total_belanja') is not None else 0
        )
    except (ValueError, Exception) as e:
        logging.error(f"Error processing file: {e}")
        return FinalOutput(
            error="Failed to process file.",
            detail=str(e)
        )

@app.post("/struk-batch", response_model=List[FinalOutput])
async def struk_batch(files: List[UploadFile] = File(...)):
    """Process a batch of receipt images."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    
    df, descriptions, data_list = load_data_pusat()
    normalizer = get_keyword_normalizer()

    loop = asyncio.get_running_loop()
    # Gunakan ThreadPoolExecutor untuk membatasi jumlah pekerjaan berat (CPU/IO) yang berjalan bersamaan
    executor = ThreadPoolExecutor(max_workers=OCR_CONCURRENCY)
    # Gunakan Semaphore untuk membatasi jumlah file yang dibaca ke memori pada satu waktu
    semaphore = asyncio.Semaphore(OCR_CONCURRENCY)

    async def _process_and_save(file: UploadFile):
        """Fungsi helper untuk memproses satu file dan menyimpannya."""
        await semaphore.acquire()
        try:
            # Baca konten file setelah mendapat 'izin' dari semaphore
            content = await file.read()
            # Jalankan proses OCR dan matching di thread terpisah agar tidak memblokir event loop
            result = await loop.run_in_executor(executor, process_single_file, content, descriptions, data_list, normalizer)
            # Jika berhasil, jalankan proses penyimpanan ke DB di thread terpisah juga
            if not result.error:
                await loop.run_in_executor(executor, save_results_to_db, result)
            return result
        finally:
            semaphore.release()

    tasks = [asyncio.create_task(_process_and_save(f)) for f in files]
    results = await asyncio.gather(*tasks)
    return results

@app.post("/add-normalization", response_model=Dict[str, str])
def add_normalization(data: NormalizationRequest):
    """Endpoint to add new normalization keywords to the DB."""
    normalizer = get_keyword_normalizer()
    normalizer.add_keyword_to_db(data.keyword, data.normalized_value)
    return {"message": f"Normalization '{data.keyword}' -> '{data.normalized_value}' successfully saved to DB."}

@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "running"}

if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run("struk:app", host="127.0.0.1", port=8000, reload=True)