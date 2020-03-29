from det_train_generator import *

import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from tracking.yolo3.model import yolo_body, yolo_loss

origin_model_path = os.path.join(model_base, 'yolo.h5')
coarse_model_path = os.path.join(model_base, 'trained_weights_stage_1.h5')


def train(is_coarse_available=False):
    classes_path = os.path.join(model_base, 'visdrone_classes.txt')
    anchors_path = os.path.join(model_base, 'yolo_anchors.txt')
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (672, 992)  # multiple of 32, hw
    if is_coarse_available:
        model_path = coarse_model_path
    else:
        model_path = origin_model_path
    model = create_model(input_shape, anchors, num_classes,
                         freeze_body=2, weights_path=model_path)  # make sure you know what you freeze

    logging = TensorBoard(log_dir=model_base)
    checkpoint = ModelCheckpoint(os.path.join(model_base, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    '''num_train = len(os.listdir(det_train_img_dir))
    num_val = len(os.listdir(det_val_img_dir))'''
    # use mot-seqs to train detector
    train_seq_ids = all_seq_ids('train')
    val_seq_ids = all_seq_ids('val')
    num_train = len(train_seq_ids)
    num_val = len(val_seq_ids)
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if not is_coarse_available:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 18
        # print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(seq_train_for_det_generator(batch_size, input_shape, anchors, num_classes,
                                                        file_ids=train_seq_ids),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=seq_val_for_det_generator(batch_size, input_shape, anchors,
                                                                      num_classes, file_ids=val_seq_ids),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=50,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint, early_stopping])
        model.save_weights(os.path.join(model_base ,'trained_weights_stage_1.h5'))

    # Unfreeze and continue training, to fine-tune.
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    model.compile(optimizer=Adam(lr=1e-6),
                  loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
    print('Unfreeze all of the layers.')

    batch_size = 2  # note that more GPU memory is required after unfreezing the body
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit_generator(seq_train_for_det_generator(batch_size, input_shape, anchors, num_classes,
                                                    file_ids=train_seq_ids),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=seq_val_for_det_generator(batch_size, input_shape, anchors,
                                                                  num_classes, file_ids=val_seq_ids),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=150,
                        initial_epoch=50,
                        callbacks=[logging, checkpoint, reduce_lr, early_stopping])
    model.save_weights(os.path.join(model_base , 'trained_weights_final.h5'))

    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, weights_path, load_pretrained=True, freeze_body=2):
    '''create the training model'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model





if __name__ == '__main__':
    train(True)
