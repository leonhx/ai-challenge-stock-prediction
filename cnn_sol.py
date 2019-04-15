import numpy as np
import pandas as pd
import tensorflow as tf

train_full = pd.read_csv('data/20170916/ai_challenger_stock_train_20170916/stock_train_data_20170916.csv')
test_full = pd.read_csv('data/20170916/ai_challenger_stock_test_20170916/stock_test_data_20170916.csv')

train_data = train_full[train_full['era'] <= 15]
val_data = train_full[train_full['era'] > 15]
test_data = test_full
raw_features = [col for col in train_full.columns if col.startswith('feature')]

def transform_dataset(dataframe, feature_cols):
    X = dataframe[feature_cols]
    y = dataframe['label']
    return X, y

X_placeholder = tf.placeholder(tf.float32, [None, len(raw_features)], name='X')
y_placeholder = tf.placeholder(tf.int32, [None], name='y')
weight_placeholder = tf.placeholder(tf.float32, [None], name='weight')

def res_block(inputs, filters, kernel_size, reuse=False, name='res'):
    xavier_norm_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
    conv1 = tf.layers.conv1d(inputs, filters, kernel_size,
                             padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=xavier_norm_initializer,
                             name='%s_conv1' % name, reuse=reuse)
    conv2 = tf.layers.conv1d(inputs, filters, kernel_size,
                             padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=xavier_norm_initializer,
                             name='%s_conv2' % name, reuse=reuse)
    conv3 = tf.layers.conv1d(inputs, filters, kernel_size,
                             padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=xavier_norm_initializer,
                             name='%s_conv3' % name, reuse=reuse)
    return tf.add(conv1, conv3)

def build_graph(training=True, reuse=False):
    X_0 = tf.expand_dims(X_placeholder, 2)
    xavier_norm_initializer = tf.contrib.layers.xavier_initializer(uniform=False)

    res1 = res_block(X_0, 16, 4, reuse=reuse, name='res1')
    maxpool1 = tf.layers.max_pooling1d(res1, 4, 4, padding='same', name='maxpool1')

    print('res-maxpool[1]:', maxpool1.shape)

    res2 = res_block(maxpool1, 16, 4, reuse=reuse, name='res2')
    maxpool2 = tf.layers.max_pooling1d(res2, 4, 4, padding='same', name='maxpool2')

    print('res-maxpool[2]:', maxpool2.shape)

    res3 = res_block(maxpool2, 16, 4, reuse=reuse, name='res3')
    maxpool3 = tf.layers.max_pooling1d(res3, 2, 2, padding='same', name='maxpool3')

    print('res-maxpool[3]:', maxpool3.shape)

    # conv4 = tf.layers.conv1d(maxpool3, 16, 4,
    #                          padding='same',
    #                          activation=tf.nn.relu,
    #                          kernel_initializer=xavier_norm_initializer,
    #                          name='conv4', reuse=reuse)
    # maxpool4 = tf.layers.max_pooling1d(conv4, 2, 2, padding='same', name='maxpool4')

    # print('conv-maxpool[4]:', maxpool4.shape)

    # conv5 = tf.layers.conv1d(maxpool4, 16, 4,
    #                          padding='same',
    #                          activation=tf.nn.relu,
    #                          kernel_initializer=xavier_norm_initializer,
    #                          name='conv5', reuse=reuse)
    # maxpool5 = tf.layers.max_pooling1d(conv5, 3, 3, padding='same', name='maxpool5')

    # print('conv-maxpool[5]:', maxpool5.shape)

    conv_result = maxpool3
    print('conv-final:', conv_result.shape)

    input_shape = conv_result.shape.as_list()
    batch_size = tf.shape(conv_result)[0]
    flatten = tf.reshape(conv_result, [batch_size, input_shape[1] * input_shape[2]])

    # fc layers
    flatten_dropout = tf.layers.dropout(flatten, rate=0.5, training=training, name='flatten_dropout')
    # fc1 = tf.layers.dense(flatten_dropout, 32,
    #                       activation=tf.nn.relu,
    #                       kernel_initializer=xavier_norm_initializer,
    #                       name='fc1', reuse=reuse)
    # fc1_dropout = tf.layers.dropout(fc1, rate=0.5, training=training, name='fc1_dropout')
    fc2 = tf.layers.dense(flatten_dropout, 2,
                          activation=tf.nn.relu,
                          kernel_initializer=xavier_norm_initializer,
                          name='fc2', reuse=reuse)
    if training:
        pred = None
    else:
        pred = tf.nn.softmax(fc2)

    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_placeholder, logits=fc2)
    loss = tf.reduce_sum(xentropy * weight_placeholder / tf.reduce_sum(weight_placeholder))

    if training:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss)
    else:
        train_op = None

    return train_op, loss, pred


train_op, train_loss, _ = build_graph()
_, val_loss, pred = build_graph(training=False, reuse=True)

sess = tf.Session()
sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())


def train(loss, train_op):
    batch_size = 64
    num_batches = train_data.shape[0] // batch_size
    my_train_data = train_data.sample(n=num_batches * batch_size).reset_index(drop=True)
    total_train_loss = 0.0
    for batch_no in range(num_batches):
        start_idx = batch_no * batch_size
        end_idx = (batch_no + 1) * batch_size
        batch = my_train_data.iloc[start_idx:end_idx]
        x_, y_ = transform_dataset(batch, raw_features)
        weight_ = batch['weight']
        train_loss, _ = sess.run([loss, train_op], feed_dict={
            X_placeholder: np.array(x_),
            y_placeholder: np.array(y_),
            weight_placeholder: np.array(weight_)
        })
        total_train_loss += train_loss
    print('average train loss: %f' % (total_train_loss / num_batches))


def validate(loss):
    x_, y_ = transform_dataset(val_data, raw_features)
    weight_ = val_data['weight']
    val_loss = sess.run(loss, feed_dict={
        X_placeholder: np.array(x_),
        y_placeholder: np.array(y_),
        weight_placeholder: np.array(weight_)
    })
    print('average val loss: %f' % val_loss)


def dump(i, pred):
    test_id = test_data['id']
    x_ = test_data[raw_features]
    test_pred = sess.run(pred, feed_dict={
        X_placeholder: np.array(x_)
    })
    proba = test_pred[:, 1]
    with open('pred_result_%d.csv' % i, 'w') as f:
        f.writelines(['id,proba\n'])
        f.writelines(['%d,%f\n' % (i, p) for i, p in zip(test_id, proba)])


def main(_):
    epochs = 10
    for i in range(epochs):
        train(train_loss, train_op)
        validate(val_loss)
        dump(i, pred)


if __name__ == '__main__':
    tf.app.run()
