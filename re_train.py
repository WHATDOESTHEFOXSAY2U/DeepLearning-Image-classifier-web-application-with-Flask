import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
import os

print("*    import Done !")

HEIGHT = 224 # VGG16 only accept Images with size of  224 X 224, so resize Input Image
WIDTH = 224

base_model = VGG16( weights='imagenet',include_top=False,input_shape=(HEIGHT, WIDTH, 3))

TRAIN_DIR = "Img_data"
HEIGHT = 224
WIDTH = 224
BATCH_SIZE = 8

train_datagen =  ImageDataGenerator(preprocessing_function=preprocess_input,
                                    rotation_range=90,
                                    horizontal_flip=True,
                                    vertical_flip=True)

train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                    target_size=(HEIGHT, WIDTH),
                                                    batch_size=BATCH_SIZE)

def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False 

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x)

    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model


class_list = ["Amy", "Isla"]
FC_LAYERS = [300, 300]
dropout = 0.5

finetune_model = build_finetune_model(base_model,
                                      dropout=dropout,
                                      fc_layers=FC_LAYERS,
                                      num_classes=len(class_list))

NUM_EPOCHS = 10
BATCH_SIZE = 8
num_train_images = 114

adam = Adam(lr=0.00001)
finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

filepath="You_Path_Where_to_save" + "VGG16" + "_model_weights.h5"
checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
callbacks_list = [checkpoint]

history = finetune_model.fit_generator(train_generator, epochs=NUM_EPOCHS, workers=1,
                                       steps_per_epoch=num_train_images // BATCH_SIZE,
                                       shuffle=True, callbacks=callbacks_list)

print("*    Re-Training Complite")
