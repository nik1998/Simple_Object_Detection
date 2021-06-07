import keras_retinanet
import keras
from keras_retinanet import models, losses
from keras_retinanet.preprocessing import pascal_voc
import tensorflow as tf
import os
#tf.compat.v1.disable_eager_execution()
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
from keras_retinanet.utils.tf_version import check_tf_version

check_tf_version()
if __name__ == '__main__':
    #files_dir = 'fruit_dataset/test/JPEGImages'
    #for image in os.listdir(files_dir):
        #print(image[:-4])
    model = keras_retinanet.models.backbone('resnet50').retinanet(num_classes=3)
    model.compile(
        loss={
            'regression': keras_retinanet.losses.smooth_l1(),
            'classification': keras_retinanet.losses.focal()
        },
        optimizer=keras.optimizers.Adam(lr=1e-5, clipnorm=0.001)
    )
    labels = {'apple':0, 'banana':1, 'orange':2}
    train_gen = keras_retinanet.preprocessing.pascal_voc.PascalVocGenerator("fruit_dataset/train", "train",
                                                                            classes=labels)
    test_gen = keras_retinanet.preprocessing.pascal_voc.PascalVocGenerator("fruit_dataset/test", "test",
                                                                     classes=labels)
    model.summary()
    #
    #batch_size = 10,
    #steps_per_epoch = 240,
    model.fit(train_gen, epochs=10,  verbose=True, validation_data = test_gen)
