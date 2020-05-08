import numpy as np
import pandas as pd
from math import floor
import tensorflow as tf
import gc
import os
import cv2
import pickle

import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from model import create_model
from utils import plot_summaries


seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)

DATA_DIR = 'D:/WPI/DS504/FinalProject/bengaliai-cv19'
TRAIN_DIR = 'D:/WPI/DS504/FinalProject/train/'

HEIGHT = 137
WIDTH = 236
SCALE_FACTOR = 0.70
HEIGHT_NEW = int(HEIGHT * SCALE_FACTOR)
WIDTH_NEW = int(WIDTH * SCALE_FACTOR)
RUN_NAME = 'Train1_'
PLOT_NAME1 = 'Train1_LossAndAccuracy.png'
PLOT_NAME2 = 'Train1_Recall.png'

BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 80
TEST_SIZE = 1./6


def ModelCheckpointFull(model_name):
    return ModelCheckpoint(model_name,
                            monitor = 'val_loss',
                            verbose = 1,
                            save_best_only = False,
                            save_weights_only = True,
                            mode = 'min',
                            period = 1)


def _read(path):
    img = cv2.imread(path)
    return img


def CustomReduceLRonPlateau(model, history, epoch):
    global best_val_loss

    monitor = 'val_root_loss'
    patience = 5
    factor = 0.75
    min_lr = 1e-5

    current_lr = float(K.get_value(model.optimizer.lr))

    print('Current LR: {0}'.format(current_lr))

    current_val_loss = history[monitor][-1]
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
    print('Best Vall Loss: {0}'.format(best_val_loss))

    if len(history[monitor]) >= patience:
        last5 = history[monitor][-5:]
        print('Last: {0}'.format(last5))
        best_in_last = min(last5)
        print('Min value in Last: {0}'.format(best_in_last))

        if best_val_loss < best_in_last:
            new_lr = current_lr * factor
            if new_lr < min_lr:
                new_lr = min_lr
            print('ReduceLRonPlateau setting learning rate to: {0}'.format(new_lr))
            K.set_value(model.optimizer.lr, new_lr)


class TrainDataGenerator(keras.utils.Sequence):
    def __init__(self, X_set, Y_set, ids, batch_size=16, img_size=(512, 512, 3), img_dir=TRAIN_DIR, *args, **kwargs):
        self.X = X_set
        self.ids = ids
        self.Y = Y_set
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.on_epoch_end()

        # Split Data
        self.x_indexed = self.X[self.ids]
        self.y_indexed = self.Y.iloc[self.ids]

        # Prep Y per Label
        self.y_root = self.y_indexed.iloc[:, 0:types['grapheme_root']].values
        self.y_vowel = self.y_indexed.iloc[:,
                       types['grapheme_root']:types['grapheme_root'] + types['vowel_diacritic']].values
        self.y_consonant = self.y_indexed.iloc[:, types['grapheme_root'] + types['vowel_diacritic']:].values

    def __len__(self):
        return int(floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X, Y_root, Y_vowel, Y_consonant = self.__data_generation(indices)
        return X, {'root': Y_root, 'vowel': Y_vowel, 'consonant': Y_consonant}

    def on_epoch_end(self):
        self.indices = np.arange(len(self.ids))

    def __data_generation(self, indices):
        X = np.empty((self.batch_size, *self.img_size))
        Y_root = np.empty((self.batch_size, 168), dtype=np.int16)
        Y_vowel = np.empty((self.batch_size, 11), dtype=np.int16)
        Y_consonant = np.empty((self.batch_size, 7), dtype=np.int16)

        for i, index in enumerate(indices):
            ID = self.x_indexed[index]
            image = _read(self.img_dir + ID + ".png")

            X[i,] = image

        Y_root = self.y_root[indices]
        Y_vowel = self.y_vowel[indices]
        Y_consonant = self.y_consonant[indices]

        return X, Y_root, Y_vowel, Y_consonant

train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
tgt_cols = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
desc_df = train_df[tgt_cols].astype('str').describe()
types = desc_df.loc['unique',:]
X_train = train_df['image_id'].values
train_df = train_df[tgt_cols].astype('uint8')
for col in tgt_cols:
    train_df[col] = train_df[col].map('{:03}'.format)
Y_train = pd.get_dummies(train_df)

del train_df
gc.collect()

model = create_model(input_shape = (HEIGHT_NEW, WIDTH_NEW, CHANNELS))

model.compile(optimizer = Adam(lr = 0.00016),
                loss = {'root': 'categorical_crossentropy',
                        'vowel': 'categorical_crossentropy',
                        'consonant': 'categorical_crossentropy'},
                loss_weights = {'root': 0.40,
                                'vowel': 0.30,
                                'consonant': 0.30},
                metrics = {'root': ['accuracy', tf.keras.metrics.Recall()],
                            'vowel': ['accuracy', tf.keras.metrics.Recall()],
                            'consonant': ['accuracy', tf.keras.metrics.Recall()] })

print(model.summary())

msss = MultilabelStratifiedShuffleSplit(n_splits = EPOCHS, test_size = TEST_SIZE, random_state = seed)

best_val_loss = np.Inf
history = {}

for epoch, msss_splits in zip(range(0, EPOCHS), msss.split(X_train, Y_train)):
    print('=========== EPOCH {}'.format(epoch))

    train_idx = msss_splits[0]
    valid_idx = msss_splits[1]
    np.random.shuffle(train_idx)
    print('Train Length: {0}   First 10 indices: {1}'.format(len(train_idx), train_idx[:10]))
    print('Valid Length: {0}    First 10 indices: {1}'.format(len(valid_idx), valid_idx[:10]))

    data_generator_train = TrainDataGenerator(X_train,
                                              Y_train,
                                              train_idx,
                                              BATCH_SIZE,
                                              (HEIGHT_NEW, WIDTH_NEW, CHANNELS),
                                              img_dir=TRAIN_DIR)
    data_generator_val = TrainDataGenerator(X_train,
                                            Y_train,
                                            valid_idx,
                                            BATCH_SIZE,
                                            (HEIGHT_NEW, WIDTH_NEW, CHANNELS),
                                            img_dir=TRAIN_DIR)

    TRAIN_STEPS = int(len(data_generator_train))
    VALID_STEPS = int(len(data_generator_val))
    print('Train Generator Size: {0}'.format(len(data_generator_train)))
    print('Validation Generator Size: {0}'.format(len(data_generator_val)))

    model.fit_generator(generator=data_generator_train,
                        validation_data=data_generator_val,
                        steps_per_epoch=TRAIN_STEPS,
                        validation_steps=VALID_STEPS,
                        epochs=1,
                        callbacks=[ModelCheckpointFull(RUN_NAME + 'model_' + str(epoch) + '.h5')],
                        verbose=1)

    temp_history = model.history.history
    if epoch == 0:
        history = temp_history
    else:
        for k in temp_history: history[k] = history[k] + temp_history[k]

    CustomReduceLRonPlateau(model, history, epoch)

    with open('history.pkl', 'wb') as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)

    del data_generator_train, data_generator_val, train_idx, valid_idx
    gc.collect()

plot_summaries(history, PLOT_NAME1, PLOT_NAME2)