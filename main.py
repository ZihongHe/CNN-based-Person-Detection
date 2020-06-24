import numpy as np
import os
import cv2
import csv
import imageio
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras import Model, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model


def read_vedio(vedio_load, stop_limit):

    capture = cv2.VideoCapture(vedio_load+'.mp4')

    frames = []

    count_frame = 0

    while 1:
        _, frame = capture.read()

        count_frame += 1
        if count_frame == stop_limit or frame is None:
            break

        frames.append(frame)

    # release video capture
    capture.release()

    return frames

def frames_as_dataset(frames, stride, width, height):

    # generate  width*height*3 objects from each frame with indicated stride

    frame_dataset = []

    num_object_height = int((frames[0].shape[0] - height) / stride + 1)
    num_object_width = int((frames[0].shape[1] - width) / stride + 1)

    for frame in frames:
        for i in range(num_object_height):
            for j in range(num_object_width):
                # the current detection object
                object = frame[i * stride:i * stride + height, j * stride:j * stride + width, :]
                frame_dataset.append(object)
    frame_dataset = np.asarray(frame_dataset)

    return frame_dataset, num_object_height, num_object_width

def preprocess(image_load, frames, width, height, test_percent, frame_dataset, object_perframe):

    # face data set reading
    face_img = []
    pointer = 0

    # preprocess
    # convert image data into numpy array
    for img_name in os.listdir(image_load):
        try:
            # convert image to array, resize to the indicated width&height, and save in dataset
            image = cv2.imread(image_load + "/" + img_name)
            image_resize = cv2.resize(image, (width, height))
            # augment the original image to generate a new image
            image_resize_horizontal = cv2.flip(image_resize, 1)
            image_resize_vertical = cv2.flip(image_resize, 0)
            image_resize_all = cv2.flip(image_resize, -1)
            face_img.append(image_resize)
            face_img.append(image_resize_horizontal)
            face_img.append(image_resize_vertical)
            face_img.append(image_resize_all)
            pointer += 1
            if pointer % 100 == 0:
                print(pointer, "of images have been processed")
        except:
            continue

    face_img = np.asarray(face_img)

    # create a non-face array based on the read in frame
    # ensure length of  non-face dataset is similar to face dataset
    if(len(face_img) / len(frame_dataset)>2):
        # if frame object non-face data is smaller, np.tile the nonface data
        times = int(len(face_img) / len(frame_dataset))
        nonface_img = np.tile(frame_dataset, (times, 1, 1, 1))
    elif(len(frame_dataset) / len(face_img)>2):
        # if frame object non-face data is bigger, choose a part of frame out as non-face data to ensure a similar length with face data
        times = int(len(frame_dataset) / len(face_img))
        object_index = np.arange(0, times, len(frames))
        nonface_img_copy = frame_dataset[0:object_perframe]
        try:
            for element in object_index[1:]:
                nonface_img_copy = np.concatenate((nonface_img_copy, frame_dataset[element * object_perframe:(element + 1) * object_perframe]))
        except:
            print("warning, given-person's face images dataset too small, please add more")

        nonface_img = nonface_img_copy
        nonface_img_copy = None
    else:
        nonface_img = frame_dataset
    frame_dataset = None


    face_label = np.ones(len(face_img))
    nonface_label = np.zeros(len(nonface_img))

    # train test set generation
    # split train and test set in face and non-face dataset, according to the test percentage
    # then concaten face and non-face dataset together, when make it random inside train/test dataset

    random_index_face = np.random.choice(len(face_img), len(face_img), replace=False)
    random_index_nonface = np.random.choice(len(nonface_img), len(nonface_img), replace=False)

    train_index_face = random_index_face[int(test_percent * len(face_img)):len(face_img)]
    test_index_face = random_index_face[0:int(test_percent * len(face_img))]
    train_index_nonface = random_index_nonface[int(test_percent * len(nonface_img)):len(nonface_img)]
    test_index_nonface = random_index_nonface[0:int(test_percent * len(nonface_img))]

    train_face_img = face_img[train_index_face]
    train_face_label = face_label[train_index_face]
    test_face_img = face_img[test_index_face]
    test_face_label = face_label[test_index_face]

    train_nonface_img = nonface_img[train_index_nonface]
    train_nonface_label = nonface_label[train_index_nonface]
    test_nonface_img = nonface_img[test_index_nonface]
    test_nonface_label = nonface_label[test_index_nonface]

    train_imgset = np.concatenate((train_face_img, train_nonface_img))
    test_imgset = np.concatenate((test_face_img, test_nonface_img))
    train_labelset = np.concatenate((train_face_label, train_nonface_label))
    test_labelset = np.concatenate((test_face_label, test_nonface_label))

    random_index_train = np.random.choice(len(train_imgset), len(train_imgset), replace=False)
    random_index_test = np.random.choice(len(test_imgset), len(test_imgset), replace=False)

    train_imgset = train_imgset[random_index_train]
    train_labelset = train_labelset[random_index_train]
    test_imgset = test_imgset[random_index_test]
    test_labelset = test_labelset[random_index_test]

    return train_imgset, train_labelset, test_imgset, test_labelset


def model_fordetection(train_imgset, train_labelset, test_imgset, test_labelset, epoch, batchsize, learning_rate, image_load):

    # sparse to to_categorical
    train_labelset = to_categorical(train_labelset)
    test_labelset = to_categorical(test_labelset)

    # model for detection define

    # input layer
    input = Input(shape=(height, width, 3))

    # 2d convolutional with pooling, relu activation
    conv_output1 = Conv2D(filters=32, strides=(2,2), kernel_size=5, activation='relu')(input)
    pool_output1 = MaxPool2D(pool_size=(2, 2))(conv_output1)
    conv_output2 = Conv2D(filters=8, strides=(2,2), kernel_size=4, activation='relu')(pool_output1)
    #pool_output2 = MaxPool2D(pool_size=(2, 2))(conv_output2)
    conv_output3 = Conv2D(filters=4, strides=(2,2), kernel_size=3, activation='relu')(conv_output2)

    # full connected
    Flatten_output = Flatten()(conv_output3)

    # Dense Prevent too-large number into softmax
    Dense_sigmoid = Dense(32, activation='sigmoid')(Flatten_output)

    # softmax layer
    output = Dense(train_labelset.shape[1], activation='softmax')(Dense_sigmoid)

    model = Model(input, output)
    optimer = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True, decay=1e-06)
    model.compile(loss='categorical_crossentropy', optimizer = optimer, metrics=['mse', 'accuracy'])
    model.fit(train_imgset, train_labelset, epochs=epoch, batch_size=batchsize)

    model.save(image_load+".hdf5")

    # model evaluation
    prediction = model.predict(test_imgset)
    score = model.evaluate(test_imgset, test_labelset)
    print("model evaluation by test set: loss: ", score[0], ", mean squared error: ", score[1], ", accuracy", score[2])

    return model

def detection(frames, frame_dataset, model, stride, width, height, num_object_width):


    for i in range(len(frames)):

        predictions = model.predict(frame_dataset[i])
        predictions = predictions[:,1]

        # Bounding box only works when an object is recognised as given person
        if (sum(np.asarray(predictions)[np.asarray(predictions) > 0.5])):
            # predict the highest face probability of given person
            bound_box_index = np.argmax(predictions)
            x = int(bound_box_index / num_object_width)
            y = bound_box_index % num_object_width

            # red bounding box for given person generation
            frames[i][x * stride : x * stride + height, y * stride,:-1] = 0
            frames[i][x * stride: x * stride + height, y * stride, -1] = 255
            frames[i][x * stride : x * stride + height, y * stride + width - 1, :-1] = 0
            frames[i][x * stride: x * stride + height, y * stride + width - 1, -1] = 255
            frames[i][x * stride, y * stride : y * stride + width - 1, :-1] = 0
            frames[i][x * stride, y * stride: y * stride + width - 1, -1] = 255
            frames[i][x * stride + height - 1, y * stride:y * stride + width, :-1] = 0
            frames[i][x * stride + height - 1, y * stride:y * stride + width, -1] = 255

        # blue and red chanels exchange to recover
        frames[i] = frames[i][...,::-1]

        print(i, "/", len(frames), "frames have been detected")

    return frames

def main(image_load, width, height, test_percent, vedio_load, epoch, batchsize, learning_rate, stride, stop_limit, whether_cover, fps):

    # load in vedio in frames
    frames = read_vedio(vedio_load, stop_limit)

    frame_dataset, num_object_height, num_object_width = frames_as_dataset(frames, stride, width, height)
    # object number in each frames
    object_perframe = num_object_height * num_object_width

    # check whether model for detection already exists
    if (os.path.isfile(image_load+".hdf5") and whether_cover == 0):
        model = load_model(image_load+".hdf5")
        print(image_load, "model exists, load in...")
    else:
        # read given person images and train test split
        train_imgset, train_labelset, test_imgset, test_labelset = preprocess(image_load, frames, width, height, test_percent, frame_dataset, object_perframe)

        # CNN model built given person classification
        model = model_fordetection(train_imgset, train_labelset, test_imgset, test_labelset, epoch, batchsize, learning_rate, image_load)

    frame_dataset = frame_dataset.reshape(len(frames), object_perframe, frame_dataset.shape[1], frame_dataset.shape[2], frame_dataset.shape[3])
    frames = detection(frames, frame_dataset, model, stride, width, height, num_object_width)

    # write detected frames into video
    imageio.mimwrite(vedio_load+'_detection.mp4', np.asarray(frames), fps = fps)

if __name__ == "__main__":

    # load in configuration from command line
    configuration = list(csv.reader(open("configuration.csv")))

    # image set of given person
    image_load = configuration[1][0]

    # size of img
    width = int(configuration[1][1])
    height = int(configuration[1][2])

    # test percentage
    test_percent = float(configuration[1][3])

    # vedio include given person
    vedio_load = configuration[1][4]

    # epoch of model training
    epoch = int(configuration[1][5])

    # batch size of model training
    batchsize = int(configuration[1][6])

    # learning rate of model
    learning_rate = float(configuration[1][7])

    # stride of detection object (bounding box)
    stride = int(configuration[1][8])

    # maximun of frames read from target vedio
    stop_limit = int(configuration[1][9])

    # whether cover the original model
    whether_cover = int(configuration[1][10])

    # fps of final generating vedio
    fps = int(configuration[1][11])

    main(image_load, width, height, test_percent, vedio_load, epoch, batchsize, learning_rate, stride, stop_limit, whether_cover, fps)


