#!/usr/bin/env python3
"""
main.py

Live Kinyarwanda Voice Assistant: Records speech, transcribes using KinyaWhisper,
matches question using NLP logic, and replies using Kinyarwanda TTS.
"""

import os
import torch
import torchaudio
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from difflib import get_close_matches
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from TTS.api import TTS

model_dir = "kinya-whisper-model"
output_file = "transcriptions.txt"
temp_audio = "live_input.wav"

# Load ASR model
print("🔍 Loading ASR model...")
asr_model = WhisperForConditionalGeneration.from_pretrained(model_dir)
processor = WhisperProcessor.from_pretrained(model_dir)
asr_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
asr_model.to(device)
print("✅ ASR model loaded!")

# Load TTS model
print("🔊 Loading TTS model...")
tts = TTS("DigitalUmuganda/Kinyarwanda_YourTTS")
print("✅ TTS model ready!")

# Question-Answer dictionary
qa_dict = {
    "muraho": "Muraho! Umeze mute?",
    "umeze ute": "Ni meza, urakoze. Nawe umeze ute?",
    "amakuru": "Amakuru ni meza. Wowe se?",
    "igitenge": "Igitenge ni imyambaro y'umuco nyarwanda ifite amabara y'igitangaza.",
    "amafaranga": "Amafaranga ni ingenzi mu buzima bwa buri munsi."
}

def get_best_match(text, qa_dict, cutoff=0.6):
    normalized = text.strip().lower()

    if normalized in qa_dict:
        return normalized

    matches = get_close_matches(normalized, qa_dict.keys(), n=1, cutoff=cutoff)
    if matches:
        print(f"🔍 Fuzzy matched '{normalized}' to '{matches[0]}'")
        return matches[0]

    for question in qa_dict.keys():
        if normalized in question:
            print(f"🔍 Partial match: '{normalized}' in '{question}'")
            return question

    return None

def record_audio(filename, duration=5, rate=16000):
    print("🎤 Please speak now (Recording for 5 seconds)...")
    audio = sd.rec(int(duration * rate), samplerate=rate, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, rate, audio)
    print("🎙️ Done recording.")

def transcribe(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    input_features = processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(device)
    with torch.no_grad():
        predicted_ids = asr_model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def speak_kinyarwanda(text):
    tts.tts_to_file(text=text, file_path="response.wav")
    os.system("start response.wav" if os.name == 'nt' else "aplay response.wav")

def main():
    with open(output_file, "a") as log:
        while True:
            record_audio(temp_audio)
            transcription = transcribe(temp_audio)
            print(f"📝 Transcription: {transcription}")
            log.write(f"User: {transcription}\n")

            match = get_best_match(transcription, qa_dict)
            if match:
                response = qa_dict[match]
            else:
                response = "Ndababariwe, sinabyumvise neza."

            print(f"🤖 Assistant: {response}")
            log.write(f"Bot: {response}\n\n")
            speak_kinyarwanda(response)

            # Option to exit
            cont = input("👉 Do you want to ask again? (y/n): ").strip().lower()
            if cont != 'y':
                break

if __name__ == "__main__":
    main()
