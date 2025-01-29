import threading
import tkinter as tk
from source.functions import recognise
import pickle
from tensorflow import keras
import torch
import os


# load speaking model
device = torch.device('cpu')
torch.set_num_threads(4)
local_file_ru = 'model_ru.pt'
local_file_en = 'model_en.pt'

if not os.path.isfile(local_file_ru):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v4_ru.pt',
                                   local_file_ru)

if not os.path.isfile(local_file_en):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/en/v3_en.pt',
                                   local_file_en)

# Загрузка существующей модели русского языка - если есть на компьютере
speak_model_ru = torch.package.PackageImporter(local_file_ru).load_pickle("tts_models", "model")

# загркзка существующей модели английского языка
speak_model_en = torch.package.PackageImporter(local_file_en).load_pickle("tts_models", "model")