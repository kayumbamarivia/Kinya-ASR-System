#!/usr/bin/env python3
"""
train.py

Fine-tunes OpenAI's Whisper model for Kinyarwanda ASR on a custom dataset.
Requires: Hugging Face Transformers, torchaudio, and datasets.
"""

import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import torchaudio
from datasets import load_dataset
from transformers import (Trainer, TrainingArguments, WhisperForConditionalGeneration,
                          WhisperProcessor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [f["labels"] for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )

        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": label_features},
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )

        batch["labels"] = labels
        return batch

# Load dataset from JSONL
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

# Load processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# Preprocessing function
def prepare_example(batch):
    audio_path = os.path.join("audio", os.path.basename(batch["audio"]))
    speech_array, sampling_rate = torchaudio.load(audio_path)
    batch["input_features"] = processor.feature_extractor(
        speech_array.squeeze().numpy(), sampling_rate=16000
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

# Map dataset
dataset = dataset.map(prepare_example)

# Load Whisper model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
# model = WhisperForConditionalGeneration.from_pretrained("./kinya-whisper-model")

# Training arguments
training_args = TrainingArguments(
    output_dir="./kinya-whisper-model",
    per_device_train_batch_size=4,
    learning_rate=1e-5,
    num_train_epochs=20,
    logging_steps=5,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    save_total_limit=2
)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.tokenizer,
    data_collator=data_collator
)

warnings.filterwarnings("ignore", message=".*use_reentrant.*")

# Train the model
trainer.train()

# Save model and processor
trainer.save_model("./kinya-whisper-model")
processor.save_pretrained("./kinya-whisper-model")
