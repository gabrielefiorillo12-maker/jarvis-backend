import os
import urllib.parse
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import edge_tts
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
client = Groq(api_key=GROQ_API_KEY)

@app.get("/")
def home():
    return {"status": "Jarvis Backend Attivo e Leggero!"}

@app.post("/chat")
async def chat_endpoint(
    text: str = Form(""),
    audio: UploadFile = File(None)
):
    user_text = text

    # Trascrizione Audio usando l'API ultra-veloce di Groq Whisper
    if audio:
        audio_bytes = await audio.read()
        with open("temp.wav", "wb") as f:
            f.write(audio_bytes)
        
        with open("temp.wav", "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=("temp.wav", file.read()),
                model="whisper-large-v3-turbo",
                language="it"
            )
            user_text = transcription.text

    system_prompt = "Sei Jarvis, un assistente IA ultra-rapido. Rispondi in italiano in modo diretto, sintetico ed efficace."
    messaggio_prompt = user_text.strip() or "Ciao!"

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

    # Sintesi vocale veloce
    audio_path = "reply.mp3"
    try:
        comm = edge_tts.Communicate(reply, "it-IT-DiegoNeural", rate="+25%")
        await comm.save(audio_path)
    except Exception:
        audio_path = None

    return {"text": reply, "user_prompt": user_text}
