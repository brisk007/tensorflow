import tensorflow as tf
import numpy as np
import math
import struct


image_pixels = 28 * 28
NUM_CLASSES = 10
iter_num = 100
batch_size = 32

def inference(images, hidden1_units, hidden2_units):
    with tf.name_scope('hidden1') as scope:
        weights1 = tf.Variable(
            tf.truncated_normal(
                            [image_pixels, hidden1_units],
                            stddev=1.0),
            name = 'weights')
        biases1 = tf.Variable(
            tf.zeros([hidden1_units]),
            name = 'biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights1) + biases1)


    with tf.name_scope('hidden2') as scope:
        weights2 = tf.Variable(
            tf.truncated_normal(
                    [hidden1_units, hidden2_units],
                    stddev=1.0),
            name = 'weights')
        biases2 = tf.Variable(
            tf.zeros([hidden2_units]),
            name = 'biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights2) + biases2)

    with tf.name_scope('softmax_linear'):
        weights3 = tf.Variable(
            tf.truncated_normal(
                [hidden2_units, NUM_CLASSES],
                stddev = 1.0),
            name = 'weights'
        )

        biases3 = tf.Variable(
            tf.zeros([NUM_CLASSES]),
            name = 'biases'
        )

        logits = tf.matmul(hidden2, weights3) + biases3
    return logits


def loss(logits, labels):
    print(labels)
    print (logits)
    return tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))


def train(loss, learning_rate):
    tf.summary.scalar('loss', loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))

trainImagesPath = 'C:\\Users\LK\\Desktop\source\\train-images.idx3-ubyte'
trainLabelsPath = 'C:\\Users\\LK\\Desktop\\source\\train-labels.idx1-ubyte'
testLabelsPath = 'C:\\Users\\LK\\Desktop\\source\\t10k-labels.idx1-ubyte'
testImagesPath = 'C:\\Users\\LK\\Desktop\\source\\t10k-images.idx3-ubyte'


def nextBatch(batch_size):
    with open(trainImagesPath, 'rb') as f1:
        image_buf = f1.read()
    with open(trainLabelsPath, 'rb') as f2:
        label_buf = f2.read()


    image_index = struct.calcsize('>IIII')
    label_index = struct.calcsize('>II')

    images, labels = [], []

    def norm_label(i):
        label = np.zeros(NUM_CLASSES)
        label[i] = 1
        return label

    index = 0
    while index < 60000:
        step = min(batch_size, 60000 - index)
        image_index = index * 784 + 16
        label_index = index * 1 + 8

        image = struct.unpack_from('>%dB' %(step * 784), image_buf, image_index)
        image = list(map(lambda x: 1.0 if x > 128 else 0.0, image))
        image = np.reshape(image, (step, 784))
        label = struct.unpack_from('>%dB' %(step * 1), label_buf, label_index)
        label = list(map(norm_label, label))
        index += batch_size
        yield image, label


def testData():
    with open(testImagesPath, 'rb') as f1:
        image_buf = f1.read()
    with open(testLabelsPath, 'rb') as f2:
        label_buf = f2.read()


    images, labels = [], []

    def norm_label(i):
        label = np.zeros(NUM_CLASSES)
        label[i] = 1
        return label


    image = struct.unpack_from('>7840000B', image_buf, 16)
    image = list(map(lambda x: 1.0 if x > 128 else 0.0, image))
    image = np.reshape(image, (10000, 784))
    label = struct.unpack_from('>10000B', label_buf, 8)
    label = list(map(norm_label, label))
    return image, label



def test():
    pass

def main():
    images_placehold = tf.placeholder(tf.float32, shape=(None, image_pixels))
    labels_placehold = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES))

    logits = inference(images_placehold, 20, 20)
    l = loss(logits, labels_placehold)
    train_op = train(l, 0.05)
    def f(arr):
        m = 0
        for i in range(0, len(arr)):
            if arr[m] < arr[i]:
                m = i
        return m

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for step in range(iter_num):
            b = 0
            x = 0
            for images_feed, label_feed in nextBatch(batch_size):
                # print (images_feed[0])
                # return
                _, loss_val ,out= sess.run(
                    [train_op, l, logits],
                    feed_dict={
                        images_placehold:images_feed,
                        labels_placehold:label_feed
                    }
                )
            images_feed, label_feed = testData()
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_placehold, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            acc= sess.run(accuracy, feed_dict={images_placehold: images_feed,
                                              labels_placehold: label_feed})


            print ('iter:%d\t loss=%.4f\t acc=%.4f' %(step, loss_val, acc))


if __name__ == '__main__':
    main()


























    pass
