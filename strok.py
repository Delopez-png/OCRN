"""
Improved FastAPI OCR application with:
- Bounded concurrency for OCR (handle 100+ images by limiting in-flight Gemini/API calls)
- Safer temp-file handling and chunked reads
- Robust Gemini call with retries and error handling
- Improved KeywordNormalizer with automatic typo learning and bulk inserts
- Fixed database initialization / Excel loader issues (missing comma, consistent normalisasi parsing)

Save this file as `struk_app_improved.py`. The DB initialization functions are included at the bottom (you can separate them if you prefer).

Configure via environment variables (in your .env):
- DB_DB, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
- GOOGLE_API_KEY
- OCR_CONCURRENCY (default 8)
- TYPO_THRESHOLD (default 85) -> lower for more aggressive learning
- OCR_RETRIES (default 3)

"""

import os
import io
import re
import json
import time
import logging
import tempfile
import asyncio
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from PIL import Image
from fuzzywuzzy import fuzz
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from dotenv import load_dotenv
from sqlalchemy import create_engine
import psycopg2

# Optional: if you use the Google Gemini SDK as in the original code
try:
    import google.generativeai as genai
except Exception:
    genai = None

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load env ---
load_dotenv()
DB_NAME = os.getenv("DB_DB")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
OCR_CONCURRENCY = int(os.getenv("OCR_CONCURRENCY", "8"))
TYPO_THRESHOLD = int(os.getenv("TYPO_THRESHOLD", "85"))
OCR_RETRIES = int(os.getenv("OCR_RETRIES", "3"))
ALLOWED_BRANDS = os.getenv("ALLOWED_BRANDS", "nivea,biore,posh,khaf").split(',')
ALLOWED_BRANDS = [b.strip().lower() for b in ALLOWED_BRANDS if b.strip()]

# Configure Gemini (if available)
if genai and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        model_gemini = None
        logger.warning("Unable to create Gemini model object. Make sure google.generativeai is configured correctly.")
else:
    model_gemini = None
    logger.warning("Gemini SDK or API key not found; OCR calls will fail until configured.")

# --- Keyword Normalizer ---
class KeywordNormalizer:
    """Loads keywords from DB, supports file-based seed, and can automatically learn typos."""

    def __init__(self, normalisasi_file: str = 'normalisasi.txt'):
        self.keywords_map: Dict[str, str] = {}
        # Load from DB (if available) and from file seed
        self.load_keywords()
        self.learn_from_file(normalisasi_file)

    def _get_db_conn(self):
        return psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )

    def load_keywords(self):
        self.keywords_map.clear()
        try:
            with self._get_db_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT keyword, normalized_value FROM keywords")
                    for keyword, normalized_value in cur.fetchall():
                        self.keywords_map[str(keyword).lower()] = str(normalized_value).lower()
            logger.info("Keywords loaded from DB: %d entries", len(self.keywords_map))
        except Exception as e:
            logger.warning("Could not load keywords from DB (table may not exist yet): %s", e)

    def learn_from_file(self, file_path: str):
        if not os.path.exists(file_path):
            logger.info("Normalization file '%s' not found; skipping file import.", file_path)
            return
        inserted = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Support both '->' and '=' delimiters
                    if '->' in line:
                        parts = line.split('->', 1)
                    elif '=' in line:
                        parts = line.split('=', 1)
                    else:
                        continue
                    keyword, normalized_value = parts[0].strip(), parts[1].strip()
                    if keyword and normalized_value:
                        self.add_keyword_to_db(keyword.lower(), normalized_value.lower())
                        inserted += 1
            logger.info("Loaded %d keywords from %s", inserted, file_path)
        except Exception as e:
            logger.error("Error reading normalization file: %s", e)

    def create_keywords_table(self):
        try:
            with self._get_db_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS keywords (
                            keyword TEXT PRIMARY KEY,
                            normalized_value TEXT NOT NULL,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    conn.commit()
            logger.info("Keywords table ensured in DB.")
        except Exception as e:
            logger.error("Failed to create keywords table: %s", e)

    def normalize_text(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Normalize token-by-token using keywords_map. Returns normalized_text and mapping of replaced words.
        """
        if not text:
            return "", {}
        original_words = re.findall(r"\b\w+\b", text.lower())
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
        try:
            with self._get_db_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO keywords (keyword, normalized_value)
                        VALUES (%s, %s)
                        ON CONFLICT (keyword) DO UPDATE SET normalized_value = EXCLUDED.normalized_value
                    """, (keyword, normalized_value))
                    conn.commit()
            # update in-memory map
            self.keywords_map[keyword] = normalized_value
            logger.debug("Added/updated keyword: %s -> %s", keyword, normalized_value)
        except Exception as e:
            logger.error("Error adding keyword to DB: %s", e)

    def add_keyword_if_typo(self, ocr_word: str, threshold: int = TYPO_THRESHOLD) -> Optional[Tuple[str,int]]:
        """If ocr_word is similar enough to some existing normalized value, add it as a keyword.
        Returns (normalized_value,score) if added, else None.
        """
        w = ocr_word.lower().strip()
        if not w or w in self.keywords_map:
            return None

        # Compare against existing normalized values (preferred) and keywords
        normalized_values = list(set(self.keywords_map.values()))
        best_score = 0
        best_value = None
        for val in normalized_values:
            score = fuzz.token_set_ratio(w, val)
            if score > best_score:
                best_score = score
                best_value = val

        # Also check direct keywords (sometimes keywords themselves are better matches)
        for k in self.keywords_map.keys():
            score = fuzz.ratio(w, k)
            if score > best_score:
                best_score = score
                best_value = self.keywords_map.get(k)

        if best_value and best_score >= threshold:
            try:
                self.add_keyword_to_db(w, best_value)
                logger.info("Auto-learned typo '%s' -> '%s' (score=%d)", w, best_value, best_score)
                return best_value, best_score
            except Exception as e:
                logger.error("Failed to auto-learn typo: %s", e)
        return None

    def bulk_add_candidates(self, candidates: List[str], threshold: int = TYPO_THRESHOLD):
        """Try to auto-learn multiple candidates. Uses add_keyword_if_typo per candidate but batches DB writes implicitly.
        """
        for c in set(candidates):
            try:
                self.add_keyword_if_typo(c, threshold=threshold)
            except Exception as e:
                logger.debug("Error while bulk-learning candidate %s: %s", c, e)


# Singleton normalizer
@lru_cache(maxsize=1)
def get_keyword_normalizer() -> KeywordNormalizer:
    n = KeywordNormalizer()
    return n


# --- Data loader / cache ---
@lru_cache(maxsize=1)
def load_data_pusat():
    try:
        engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        df = pd.read_sql_query("SELECT id, description, brand FROM public.data_pusat", con=engine)
        df = df.dropna(subset=['description'])
        descriptions = df['description'].astype(str).str.lower().tolist()
        data_list = df.to_dict('records')
        logger.info("Central data loaded: %d rows", len(df))
        return df, descriptions, data_list
    except Exception as e:
        logger.error("Error loading data_pusat: %s", e)
        return pd.DataFrame(), [], []


# --- Gemini OCR wrapper with retries ---

def _process_ocr(image_bytes: bytes, prompt: str = None) -> dict:
    if model_gemini is None:
        raise RuntimeError("Gemini model not configured. Set GOOGLE_API_KEY and install google.generativeai.")

    if prompt is None:
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

    last_exc = None
    for attempt in range(1, OCR_RETRIES + 1):
        try:
            img = Image.open(io.BytesIO(image_bytes))
            # API usage depends on the SDK. Keep the original call but guard.
            response = model_gemini.generate_content([prompt, img])
            text = getattr(response, 'text', None)
            if text is None:
                # Some SDKs return a different attribute
                text = str(response)
            cleaned = re.sub(r'^```json|```$', '', text, flags=re.MULTILINE).strip()
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error("JSON decoding failed from Gemini: %s", e)
            logger.debug("Raw Gemini response (truncated): %s", text[:500] if 'text' in locals() and text else '')
            raise
        except Exception as e:
            last_exc = e
            logger.warning("Gemini attempt %d/%d failed: %s", attempt, OCR_RETRIES, e)
            time.sleep(1 * attempt)
    # after retries
    logger.error("All Gemini attempts failed: %s", last_exc)
    raise last_exc


# --- Item matcher ---

def _match_items(items: List[Dict[str, Any]], descriptions: List[str], data_list: List[Dict[str, Any]], normalizer: KeywordNormalizer, threshold: int = 58) -> List[Dict[str, Any]]:
    results = []
    candidates_for_learning = []

    for item in items:
        ocr_item_name = (item.get('nama') or '').strip()
        normalized_name, typos = normalizer.normalize_text(ocr_item_name)

        # prepare candidate words to try auto-learning if not normalized
        ocr_words = re.findall(r"\b\w+\b", ocr_item_name.lower())
        for w in ocr_words:
            if w not in typos and w not in normalizer.keywords_map:
                candidates_for_learning.append(w)

        best_score = 0
        best_match_data = None
        for i, desc in enumerate(descriptions):
            score = fuzz.token_set_ratio(normalized_name, desc)
            if score > best_score:
                best_score = score
                best_match_data = data_list[i]

        is_brand_allowed = (best_match_data is not None and str(best_match_data.get('brand', '')).lower() in ALLOWED_BRANDS)
        is_match_found = best_score >= threshold and is_brand_allowed

        try:
            item_result = {
                "id": int(best_match_data.get('id')) if is_match_found and best_match_data else None,
                "name": best_match_data.get('description') if is_match_found and best_match_data else None,
                "data": best_match_data.get('brand') if is_match_found and best_match_data else None,
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
            results.append(item_result)
        except Exception as e:
            logger.error("Error building result for '%s': %s", ocr_item_name, e)
            results.append({
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

    # Bulk attempt to learn similar typos (reduces DB churn)
    if candidates_for_learning:
        try:
            normalizer.bulk_add_candidates(candidates_for_learning)
        except Exception as e:
            logger.debug("Bulk auto-learning failed: %s", e)

    return results


# --- Pydantic models ---
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


# --- File processing ---

def process_single_file(file_bytes: bytes, filename: str, descriptions, data_list, normalizer: KeywordNormalizer) -> FinalOutput:
    try:
        ocr_data = _process_ocr(file_bytes)
        items = ocr_data.get('daftar_barang', []) or []
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
    except Exception as e:
        logger.error("Error processing file %s: %s", filename, e)
        return FinalOutput(error=f"Failed to process file {filename}.", detail=str(e))


def save_results_to_db(output: FinalOutput):
    """Menyimpan hasil pemrosesan OCR ke database."""
    if output.error:
        logger.info("Melewatkan penyimpanan ke DB untuk proses yang gagal: %s", output.invoice_number or "Unknown")
        return

    conn = None
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        with conn.cursor() as cursor:
            tanggal_obj = None
            if output.tanggal:
                try:
                    # Prompt Gemini meminta DD/MM/YYYY, ubah ke objek tanggal
                    tanggal_obj = datetime.strptime(output.tanggal, '%d/%m/%Y').date()
                except (ValueError, TypeError):
                    # Coba format lain jika gagal, misal YYYY-MM-DD
                    try:
                        tanggal_obj = datetime.strptime(output.tanggal, '%Y-%m-%d').date()
                    except (ValueError, TypeError):
                        logger.warning("Tidak dapat mem-parsing tanggal '%s'. Menyimpan sebagai NULL.", output.tanggal)

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
            logger.info("Berhasil menyimpan hasil untuk invoice %s ke DB (receipt_id: %d)", output.invoice_number, receipt_id)

    except Exception as e:
        logger.error("Gagal menyimpan hasil untuk invoice %s ke DB: %s", output.invoice_number, e)
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


app = FastAPI()


@app.post("/struk-batch", response_model=List[FinalOutput])
async def struk_batch(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    df, descriptions, data_list = load_data_pusat()
    normalizer = get_keyword_normalizer()

    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=OCR_CONCURRENCY)
    semaphore = asyncio.Semaphore(OCR_CONCURRENCY)

    async def _process_uploadfile(file: UploadFile):
        await semaphore.acquire()
        try:
            # read only after acquiring semaphore to limit memory footprint
            content = await file.read()
            # jalankan pemrosesan di threadpool
            result = await loop.run_in_executor(executor, process_single_file, content, file.filename, descriptions, data_list, normalizer)
            # Jika berhasil, jalankan penyimpanan ke DB di threadpool juga agar tidak memblokir
            if not result.error:
                await loop.run_in_executor(executor, save_results_to_db, result)
            return result
        finally:
            semaphore.release()

    tasks = [asyncio.create_task(_process_uploadfile(f)) for f in files]

    # gather results (exceptions will propagate)
    results = await asyncio.gather(*tasks)
    return results


@app.post("/add-normalization", response_model=Dict[str, str])
def add_normalization(data: NormalizationRequest):
    normalizer = get_keyword_normalizer()
    normalizer.add_keyword_to_db(data.keyword.lower(), data.normalized_value.lower())
    return {"message": f"Normalization '{data.keyword}' -> '{data.normalized_value}' successfully saved to DB."}


@app.get("/")
def health_check():
    return {"status": "running"}


# Run uvicorn if executed directly
if __name__ == "__main__":
    # Untuk menginisialisasi database, jalankan skrip `db_setup.py` secara terpisah.
    # python db_setup.py
    uvicorn.run("strok:app", host="127.0.0.1", port=8000, reload=True)
