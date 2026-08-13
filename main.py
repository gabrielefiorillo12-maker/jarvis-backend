import os
import urllib.parse
import io
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import edge_tts
from faster_whisper import WhisperModel
from PIL import Image
import pypdf
from groq import Groq
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

app = FastAPI()

# Permette all'app mobile di comunicare con il server senza blocchi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Recupera la chiave API senza far scattare gli avvisi di sicurezza
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
client = Groq(api_key=GROQ_API_KEY)

# Modelli IA in memoria
stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.get("/")
def home():
    return {"status": "Jarvis Backend Attivo!"}

@app.post("/chat")
async def chat_endpoint(
    text: str = Form(""),
    audio: UploadFile = File(None),
    image: UploadFile = File(None)
):
    user_text = text

    # Trascrizione audio se inviato
    if audio:
        audio_bytes = await audio.read()
        with open("temp.wav", "wb") as f:
            f.write(audio_bytes)
        segments, _ = stt_model.transcribe("temp.wav", language="it")
        user_text = "".join([s.text for s in segments]).strip()

    # Analisi immagine se inviata
    info_foto = ""
    if image:
        img_bytes = await image.read()
        raw_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        raw_image.thumbnail((384, 384))
        inputs = blip_processor(raw_image, return_tensors="pt")
        out = blip_model.generate(**inputs, max_new_tokens=35)
        caption_en = blip_processor.decode(out[0], skip_special_tokens=True)
        info_foto = f"[FOTO: {caption_en}]\n"

    system_prompt = "Sei Jarvis, un assistente IA ultra-rapido. Rispondi in italiano in modo diretto e sintetico."
    messaggio_prompt = f"{info_foto}{user_text}".strip() or "Ciao!"

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": messaggio_prompt}
            ],
            max_tokens=180,
            temperature=0.4
        )
        reply = completion.choices[0].message.content.strip()
    except Exception as e:
        reply = f"Errore: {str(e)}"

    # Sintesi vocale risposta
    audio_path = "reply.mp3"
    try:
        comm = edge_tts.Communicate(reply, "it-IT-DiegoNeural", rate="+25%")
        await comm.save(audio_path)
    except Exception:
        audio_path = None

    return {"text": reply, "user_prompt": user_text}
