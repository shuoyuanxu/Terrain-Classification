import glob
import os
import tensorflow as tf
import numpy as np
import time
import cv2

w = 28
h = 28
# 各层参数
# 卷积层1
filter_size1 = 5  # Convolution filters are 5 x 5 pixels.
num_filters1 = 16  # There are 16 of these filters.
# 卷积层2.
filter_size2 = 5  # Convolution filters are 5 x 5 pixels.
num_filters2 = 36  # There are 36 of these filters.
# 全连接层.
fc_size = 128  # Number of neurons in fully-connected layer.
# 输入图片分辨率
img_size = 28
# 图片数据被存储在一维数组中
img_size_flat = img_size * img_size
# 重构一维数组得到图片
img_shape = (img_size, img_size)
# 灰度图只有单通道
num_channels = 1
# 分类十种数据
num_classes = 2
# 读取图片
positive_path = 'F:/image_training/p28'
negative_path = 'F:/image_training/n28'


def read_img(p_path, n_path):
    p_images = []
    for filename in os.listdir(p_path):
        if filename.endswith('.jpg'):
            filename = p_path + '/' + filename
            img = cv2.imread(filename, 0)
            p_images.append(img)
    n_images = []
    for filename in os.listdir(n_path):
        if filename.endswith('.jpg'):
            filename = n_path + '/' + filename
            img = cv2.imread(filename, 0)
            n_images.append(img)
    a = []
    i = 0
    j = 0
    for item in p_images:
        a.append(p_images[i])
        i += 1
    for item in n_images:
        a.append(n_images[j])
        j += 1
    b = []
    i = 0
    j = 0
    for item in p_images:
        b.append(1)
        i += 1
    for item in n_images:
        b.append(0)
        j += 1
    dat = np.array(a, np.float32)
    data_raw = dat.reshape((1, -1, 28, 28)).reshape((-1, 28, 28, 1))
    label_raw = np.array(b, np.float32)
    return data_raw, label_raw


# 打乱顺序
data, label = read_img(positive_path, negative_path)
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]
x_train = data
y_train = label


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
    # 从截断的正态分布中输出随机值,生成的值服从具有指定平均值mean和标准偏差stddev的正态分布


def new_biases(length):
    # equivalent to y intercept
    # constant value carried over across matrix math
    return tf.Variable(tf.constant(0.05, shape=[length]))
    # 创建一个常量tensor, 按照给出的value来赋值, 可以是一个数也可以是一个list, shape决定形状


def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
    # Filter Weight 尺寸, 横竖相等
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    # 创建卷积运算, layer即为卷积计算输出
    # [Image number, y轴， x轴， 通道数]
    # Padding 用0补齐卷积造成的图片减小让输入输出尺寸一致
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    # 偏差相加
    layer += biases
    # 是否池化
    if use_pooling:  # 2x2最大池化（输入, 池化窗口大小, 池化在每一个维度上滑动步长, padding）
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 激活函数RELU, 建议在池化之前使用, 此案例特殊处理以节省计算量
    layer = tf.nn.relu(layer)
    # 输出输出层数据以及权重张量
    return layer, weights


# 展开运算, 将卷积层输出四维张量（二维输出 图片编号 通道数）展开成全连接层二维矩阵（图片编号 特征值）
def flatten_layer(layer):
    layer_shape = layer.get_shape()
    # 输入层形式 layer_shape == [num_images, img_height, img_width, num_channels]
    # TensorFlow 函数计算特征值数目 img_height * img_width * num_channels
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])
    # 现在形式[num_images, img_height * img_width * num_channels]
    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


# 创建新的全连接层
def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    # 根据输入形式创建随机初始权重与偏差
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    # 计算输出
    layer = tf.matmul(input, weights) + biases
    # 激励函数ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


# 定义一个函数，按批次取数据
def mini_batches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


# -----------------构建网络----------------------
# 占位符
x = tf.placeholder(tf.float32, shape=[None, w, h, 1], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')
# 搭建卷积神经网络
layer_conv1, weights_conv1 = \
    new_conv_layer(input=x,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)
layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)
layer_flat, num_features = flatten_layer(layer_conv2)
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)
# ---------------------------网络结束---------------------------

loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=layer_fc2)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(layer_fc2, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 训练和测试数据，可将n_epoch设置更大一些

n_epoch = 1
batch_size = 64
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for epoch in range(n_epoch):
    start_time = time.time()

    # training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in mini_batches(x_train, y_train, batch_size, shuffle=True):
        _x, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err
        train_acc += ac
        n_batch += 1
        # print(_x)
    print("   train loss: %f" % (train_loss / n_batch))
    print("   train acc: %f" % (train_acc / n_batch))

    # # validation
    # val_loss, val_acc, n_batch = 0, 0, 0
    # for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
    #     err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
    #     val_loss += err
    #     val_acc += ac
    #     n_batch += 1
    # print("   validation loss: %f" % (val_loss / n_batch))
    # print("   validation acc: %f" % (val_acc / n_batch))
# saver.save(sess, './model.ckpt')
print(sess.run(layer_conv2))
sess.close()
