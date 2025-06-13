import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.models import load_model

def pad_sequences(data, maxlen=16000, value=0):
    ori_len = len(data)
    if len(data) < maxlen:
        final_data = [0] * (maxlen - ori_len) + list(data)
        data = final_data
    return get_spectrogram(data)


def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

path = Path("data/mini_speech_commands_extracted/mini_speech_commands")
all_file = []
for root, dirs, files in os.walk("./data/mini_speech_commands_extracted/mini_speech_commands"):
    for file in files:
        if file.endswith(".wav"):
            all_file.append(os.path.join(root, file))


labels = [f.name for f in path.iterdir() if f.is_dir()]

label_key = {val: index for index, val in enumerate(labels)}
key_label = {index: val for index, val in enumerate(labels)}

print(label_key)

all_num = 200
r_num = 0

for i in range(all_num):

    file_r = random.choice(all_file)
    model = load_model("m.keras")
    input_data = pad_sequences(sf.read(file_r)[0])
    result = model.predict(np.array([input_data]))

    real_label = file_r.split("/")[-2]

    if key_label[np.argmax(result)] == real_label:
        r_num += 1

    # print(file_r)
    # print(np.argmax(result))
    # print(key_label[np.argmax(result)])
    # print("=========================================")

print("acc", r_num / all_num)