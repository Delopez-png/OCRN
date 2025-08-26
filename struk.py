import os
import pandas as pd
import psycopg2
from fuzzywuzzy import fuzz
from PIL import Image
import google.generativeai as genai
import json
import re
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import nest_asyncio
import uvicorn
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def load_data_pusat():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    df = pd.read_sql_query("SELECT * FROM data_pusat", conn)
    conn.close()

    df = df.dropna()
    descriptions = df["Description"].astype(str).str.lower().tolist()
    ids = df["id"].tolist()
    prices = df["Harga Jual ke Konsumen yg Disarankan"].tolist()

    return df, descriptions, ids, prices

genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel("gemini-2.5-flash")

def OCR_Gemini(image_path):
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

def match_items(items, descriptions, ids, prices, threshold=58):
    result = []

    for item in items:
        nama_item = item['nama'].lower()
        best_score = 0
        best_idx = None

        for i, desc in enumerate(descriptions):
            score = fuzz.token_set_ratio(nama_item, desc)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= threshold:
            result.append({
                "id": int(ids[best_idx]),
                "name": descriptions[best_idx],
                "ocr_result": {
                    "name": item['nama'],
                    "quantity": float(item['qty']),
                    "price": float(item['harga_satuan']),
                    "total": float(item['subtotal']),
                    "accuration": round(best_score / 100, 4)
                }
            })
        else:
            result.append({
                "id": None,
                "name": None,
                "ocr_result": {
                    "name": item['nama'],
                    "quantity": float(item['qty']),
                    "price": float(item['harga_satuan']),
                    "total": float(item['subtotal']),
                    "accuration": round(best_score / 100, 4)
                }
            })
    return result


class OCRResult(BaseModel):
    name: str | None
    quantity: float | None
    price: float | None
    total: float | None
    accuration: float | None

class ItemMatched(BaseModel):
    id: int | None
    name: str | None
    ocr_result: OCRResult

class Merchant(BaseModel):
    name: str | None
    address: str | None
    phone: str | None
    email: str | None

class FinalOutput(BaseModel):
    invoice_number: str | None
    tanggal: str | None
    merchant: Merchant
    items: List[ItemMatched]
    grand_total: float | None


app = FastAPI()
df, descriptions, ids, prices = load_data_pusat()

@app.post("/struk-batch", response_model=List[FinalOutput])
async def struk_batch(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            path = tmp.name

        try:
            OCRData = OCR_Gemini(path)
            items = OCRData['daftar_barang']
            matched = match_items(items, descriptions, ids, prices)

            result = {
                "invoice_number": OCRData['invoice_number'],
                "tanggal": OCRData['tanggal'],
                "merchant": {
                    "name": OCRData['nama_toko'],
                    "address": OCRData['alamat'],
                    "phone": OCRData['phone'],
                    "email": OCRData['email']
                },
                "items": matched,
                "grand_total": float(OCRData['total_belanja'])
            }
            results.append(result)
        except Exception as e:
            results.append({"error": str(e)})

    return results

@app.get("/")
def health_check():
    return {"status": "running"}

if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run("struk:app", host="127.0.0.1", port=8000, reload=True)
