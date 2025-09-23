from telebot import TeleBot
import pickle
import librosa
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
import os

load_dotenv()


bot = TeleBot(token=os.getenv("TOKEN"))

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


@bot.message_handler(commands=['start'])
def start_bot(message):
    bot.send_message(message.chat.id, 'Send voice message!')


def audio_to_mel_spectrogram(file_path, sr=16000, n_mels=128, max_len=10):
    y, _ = librosa.load(file_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # (n_mels, time)
    log_mel_spec = log_mel_spec.T
    if log_mel_spec.shape[0] < max_len:
        pad_width = max_len - log_mel_spec.shape[0]
        log_mel_spec = np.pad(log_mel_spec, ((0, pad_width), (0, 0)))
    else:
        log_mel_spec = log_mel_spec[:max_len, :]
    return log_mel_spec

def predict_(voice_file):
    voice = audio_to_mel_spectrogram(voice_file)
    voice = voice.reshape(1, 10, 128, 1)
    y_pred = model.predict(voice)
    return np.argmax(y_pred)

@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    file_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    file_path = f"voice/{message.voice.file_id}.ogg"
    with open(file_path, "wb") as f:
        f.write(downloaded_file)
    data, samplerate = sf.read(file_path)
    sf.write("voice/output.wav", data, samplerate)
    result = predict_('voice/output.wav')
    bot.send_message(message.chat.id, f"Result: {result}")


bot.polling()