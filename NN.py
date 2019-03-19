import tensorflow as tf

def get_logits(input_images, num_results, keep_prob):
    #keep_prob = tf.placeholder(tf.float32)
    conv1_W = tf.get_variable(shape=[3, 3 ,3, 64], name='conv1_W',
        initializer=tf.contrib.layers.xavier_initializer())
    conv1_b = tf.Variable(tf.zeros([64], dtype=tf.float32),
        name='conv1_b')
    conv1 = tf.nn.conv2d(input_images, conv1_W, strides=[1, 2, 2, 1],
                         padding='SAME', name='conv1')
    # (batch size, 50, 50, 64)
    output1 = tf.nn.relu(conv1 + conv1_b, name='output1')
    output1 = tf.nn.max_pool(output1, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1], padding='SAME')
    # (batch size, 25, 25, 64)

    conv2_W = tf.get_variable(shape=[3, 3, 64, 64], name='conv2_W',
         initializer=tf.contrib.layers.xavier_initializer())
    conv2_b = tf.Variable(tf.zeros([64], dtype=tf.float32), name='conv2_b',
        dtype=tf.float32)
    conv2 = tf.nn.conv2d(output1, conv2_W, strides=[1, 2, 2, 1],
        padding='SAME', name='conv2')
    # (batch size, 13, 13, 64)
    output2 = tf.nn.relu(conv2 + conv2_b, name='output2')
    output2 = tf.nn.max_pool(output2, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='VALID')
    # (batch size, 6, 6, 64)

    conv3_W = tf.get_variable(shape=[3, 3, 64, 128], name='conv3_W',
         initializer=tf.contrib.layers.xavier_initializer())
    conv3_b = tf.Variable(tf.zeros([128], dtype=tf.float32), name='conv3_b',
        dtype=tf.float32)
    conv3 = tf.nn.conv2d(output2, conv3_W, strides=[1, 1, 1, 1],
        padding='VALID', name='conv3')
    # (batch size, 4, 4,128)
    output3 = tf.nn.relu(conv3 + conv3_b, name='output3')


    flat_output3_size = 4 * 4 * 128
    flat_output3 = tf.reshape(output3, [-1, flat_output3_size], name='flat_output3')

    fc1_W = tf.get_variable(shape=[flat_output3_size, 512], name='fc1_W',
         initializer=tf.contrib.layers.xavier_initializer())
    fc1_b = tf.Variable(tf.zeros([512], dtype=tf.float32), name='fc1_b')
    # (batch size, 256)
    output3 = tf.nn.relu(tf.matmul(flat_output3, fc1_W) + fc1_b, name='output3')
    output3 =tf.nn.dropout(output3,keep_prob)

    fc2_W = tf.get_variable(shape=[512, num_results], name='fc2_W',
                            initializer=tf.contrib.layers.xavier_initializer())
    fc2_b = tf.Variable(tf.zeros([num_results], dtype=tf.float32), name='fc2_b')
    # (batch size, num_results)
    policy_network = tf.nn.relu(tf.matmul(output3, fc2_W) + fc2_b, name='policy_network')
    policy_network = tf.nn.dropout(policy_network, keep_prob)
    return policy_network

# huber loss损失函数
def huber_loss(y_true, y_pred, max_grad=1.):

    a = tf.abs(y_true - y_pred)
    less_than_max = 0.5 * tf.square(a)
    greater_than_max = max_grad * (a - 0.5 * max_grad)
    return tf.reduce_mean(tf.where(a <= max_grad, x=less_than_max, y=greater_than_max))

#sotfmax 损失函数
def softmax_loss(logits,lables):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=lables))


def get_train_op(loss, learning_rate):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_op

