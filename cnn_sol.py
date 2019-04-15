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

sess = tf.Session()

X_0 = tf.expand_dims(X_placeholder, 2)
xavier_norm_initializer = tf.contrib.layers.xavier_initializer(uniform=False)

# conv1 = tf.layers.conv1d(X_0, 4, 8,
#                          padding='same',
#                          activation=tf.nn.relu,
#                          kernel_initializer=xavier_norm_initializer,
#                          name='conv1')
# maxpool1 = tf.layers.max_pooling1d(conv1, 2, 2, padding='same', name='maxpool1')

#conv2 = tf.layers.conv1d(maxpool1, 8, 8, padding='same', activation=tf.nn.relu, name='conv2')
#maxpool2 = tf.layers.max_pooling1d(conv2, 2, 2, padding='same', name='maxpool2')

#conv3 = tf.layers.conv1d(maxpool2, 16, 8, padding='same', activation=tf.nn.relu, name='conv3')
#maxpool3 = tf.layers.max_pooling1d(conv3, 2, 2, padding='same', name='maxpool3')

#conv4 = tf.layers.conv1d(maxpool3, 32, 8, padding='same', activation=tf.nn.relu, name='conv4')
#maxpool4 = tf.layers.max_pooling1d(conv4, 2, 2, padding='same', name='maxpool4')

#conv5 = tf.layers.conv1d(maxpool4, 64, 4, padding='same', activation=tf.nn.relu, name='conv5')
#maxpool5 = tf.layers.max_pooling1d(conv5, 2, 2, padding='same', name='maxpool5')

#flatten = tf.reshape(maxpool5, [batch_size, -1])
# flatten = tf.reshape(maxpool1, [batch_size, -1])
flatten = tf.reshape(X_0, [tf.shape(X_0)[0], -1])

fc1 = tf.layers.dense(flatten, 32,
                      activation=tf.nn.relu,
                      kernel_initializer=xavier_norm_initializer,
                      name='fc1')
fc2 = tf.layers.dense(fc1, 2, kernel_initializer=xavier_norm_initializer, name='fc2')
pred = tf.nn.softmax(fc2)

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_placeholder, logits=fc2)
weighted_loss = xentropy * weight_placeholder
loss = tf.reduce_mean(weighted_loss)

optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())

def train():
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


def validate():
    x_, y_ = transform_dataset(val_data, raw_features)
    weight_ = val_data['weight']
    val_loss = sess.run(loss, feed_dict={
        X_placeholder: np.array(x_),
        y_placeholder: np.array(y_),
        weight_placeholder: np.array(weight_)
    })
    print('average val loss: %f' % val_loss)


def dump(i):
    test_id = test_data['id']
    x_ = test_data[raw_features]
    test_pred = sess.run(pred, feed_dict={
        X_placeholder: np.array(x_)
    })
    print(test_pred)


def main(_):
    epochs = 1
    for _ in range(epochs):
        train()
        validate()
    dump(0)


if __name__ == '__main__':
    tf.app.run()
