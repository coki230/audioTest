from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
import matplotlib.pyplot as plt

# load the voice data
path = Path("data/mini_speech_commands_extracted/mini_speech_commands")
labels = [f.name for f in path.iterdir() if f.is_dir()]
print(labels)

label_key = {val: index for index, val in enumerate(labels)}
key_label = {index: val for index, val in enumerate(labels)}

# get the audio data
def get_audio(path):
    for f in path.iterdir():
        if f.is_dir():
            label = f.name
            data = []
            data_labels = []
            for file in f.iterdir():
                # add the audio data and the label
                data.append(np.array(pad_sequences(sf.read(file)[0])))
                data_labels.append(label_hot_encoder(label))
            # data = tf.keras.preprocessing.sequence.pad_sequences(data, maxlen=16000, value=0)
            yield data, data_labels

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

def get_all_data(path):
    data = []
    data_labels = []
    for d, l in get_audio(path):
        data.extend(d)
        data_labels.extend(l)
    return data, data_labels

def label_hot_encoder(label):
    label_ret = np.zeros(8)
    label_ret[label_key[label]] = 1
    return label_ret
# ================================================test============================================================
# def draw_data(data):
#     plt.subplots(3,4, figsize=(12, 8))
#     for i in range(12):
#         plt.subplot(3,4,i+1)
#         plt.plot(data[i])
#     plt.show()
#
# draw_data(next(get_audio(path))[0])

# gen_get_audio = get_audio(path)
# for a in gen_get_audio:
#     print(a)

print(np.array(next(get_audio(path))[0]).shape)
# ================================================test============================================================

# define the model
print("len of labels", len(labels))


model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(124, 129, 1)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(8, activation='softmax')

])

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=3, verbose=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, restore_best_weights=True)
callbacks = [reduce_lr, early_stop]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy'],
)

train_data = get_all_data(path)
# shuffle the data
shuffle_index = np.random.permutation(len(train_data[0]))
t_data = np.array(train_data[0])[shuffle_index]
t_label = np.array(train_data[1])[shuffle_index]
history =model.fit(
    t_data,
    t_label,
    validation_split=0.2,
    batch_size=32,
    epochs=100,
    callbacks=callbacks
)

print(history.history)

model.save("m.keras", include_optimizer=False)

# # load the model
# saved_model = tf.keras.models.load_model("m.keras")
#
# voice_file = sf.read("data/mini_speech_commands_extracted/mini_speech_commands/right/0c2ca723_nohash_0.wav")
#
# voice_file = [voice_file[0]]
#
# voice_file = tf.keras.preprocessing.sequence.pad_sequences(voice_file, padding='post', maxlen=16000, value=0)
# # make a prediction
# prediction = saved_model.predict(voice_file)
# print(prediction)
# print(key_label[np.argmax(prediction, axis=-1)[0]])

