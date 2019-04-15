"""CNN solution implemented by model_fn."""

import numpy as np
import pandas as pd
import tensorflow as tf


def xavier_norm_initializer(uniform=False):
    return tf.contrib.layers.xavier_initializer(uniform=uniform)


def conv1d(inputs, filters, kernel_size, name=None, reuse=None):
    return tf.layers.conv1d(inputs, filters, kernel_size, padding='same',
                            activation=tf.nn.relu,
                            kernel_initializer=xavier_norm_initializer(),
                            name=name, reuse=reuse)


def max_pooling1d(inputs, pool_size, strides, name=None):
    return tf.layers.max_pooling1d(inputs, pool_size, strides, padding='same', name=name)


def res_block(inputs, filters, kernel_size, reuse, name):
    conv1 = conv1d(inputs, filters, kernel_size, name='%s_conv1' % name, reuse=reuse)
    conv2 = conv1d(conv1, filters, kernel_size, name='%s_conv2' % name, reuse=reuse)
    conv3 = conv1d(conv2, filters, kernel_size, name='%s_conv3' % name, reuse=reuse)
    return tf.add(inputs, conv3, name='%s_merge' % name)


def conv_net(x_dict, params, reuse, is_training):
    raw_features = x_dict['raw']
    # group = x_dict['group']

    inputs = tf.expand_dims(raw_features, 2)

    res1 = res_block(inputs, 16, 4, reuse=reuse, name='res1')
    res1 = max_pooling1d(res1, 4, 4, name='maxpool1')
    print('res-maxpool1:', res1.shape)

    res2 = res_block(res1, 16, 4, reuse=reuse, name='res2')
    res2 = max_pooling1d(res2, 4, 4, name='maxpool2')
    print('res-maxpool2:', res2.shape)

    res3 = res_block(res2, 16, 4, reuse=reuse, name='res3')
    res3 = max_pooling1d(res3, 2, 2, name='maxpool3')
    print('res-maxpool3:', res3.shape)

    res4 = res_block(res3, 16, 4, reuse=reuse, name='res4')
    res4 = max_pooling1d(res4, 2, 2, name='maxpool4')
    print('res-maxpool4:', res4.shape)

    res5 = res_block(res4, 16, 4, reuse=reuse, name='res5')
    res5 = max_pooling1d(res5, 2, 2, name='maxpool5')
    print('res-maxpool5:', res5.shape)

    conv_result = res5
    print('conv-final:', conv_result.shape)

    result_shape = conv_result.shape.as_list()
    batch_size = tf.shape(conv_result)[0]
    flatten = tf.reshape(conv_result, [batch_size, result_shape[1] * result_shape[2]])

    flatten_dropout = tf.layers.dropout(
        flatten, rate=0.5, training=is_training, name='flatten_dropout')
    out = tf.layers.dense(flatten_dropout, 2, activation=tf.nn.relu,
                          kernel_initializer=xavier_norm_initializer(),
                          name='fc1', reuse=reuse)

    return out


def model_fn(features, labels, mode, params):
    logits_train = conv_net(features, params, reuse=False, is_training=True)
    logits_test = conv_net(features, params, reuse=True, is_training=False)

    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_probas)

    weights = features['weight']
    loss_op = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits_train) * weights / tf.reduce_sum(weights))

    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    acc_op = tf.metrics.accuracy(
        labels=labels, predictions=pred_classes, weights=weights)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_probas,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})


def dump(i, test_ids, probas):
    len_test_ids = len(test_ids)
    len_probas = len(probas)
    if len_test_ids != len_probas:
        raise AssertionError('size not match: %d /= %d' % (len_test_ids, len_probas))
    with open('pred_result_%d.csv' % i, 'w') as f:
        f.writelines(['id,proba\n'])
        f.writelines(['%d,%f\n' % (i, p) for i, p in zip(test_ids, probas)])


def main(_):
    model_seed = np.random.randint(1024)
    print('model random seed: %d' % model_seed)

    model_conf = tf.estimator.RunConfig().replace(tf_random_seed=model_seed)
    model_params = {'learning_rate': 0.001}
    model = tf.estimator.Estimator(model_fn, config=model_conf, params=model_params)

    version = '20170923'
    train_full = pd.read_csv(
        'data/{0}/ai_challenger_stock_train_{0}/stock_train_data_{0}.csv'.format(version))
    test_full = pd.read_csv(
        'data/{0}/ai_challenger_stock_test_{0}/stock_test_data_{0}.csv'.format(version))

    era_th = 16
    train_data = train_full[train_full['era'] <= era_th]
    eval_data = train_full[train_full['era'] > era_th]
    test_data = test_full

    train_seed = np.random.randint(1024)
    print('train random seed: %d' % train_seed)
    train_data = train_data.sample(n=train_data.shape[0], random_state=train_seed)

    epochs = 3
    for i in range(epochs):
        model.train(tf.estimator.inputs.numpy_input_fn(
            x=get_input_x(train_data), y=get_input_y(train_data),
            batch_size=64, shuffle=False))

        print('[%d] Train metrics:' % i, model.evaluate(
            tf.estimator.inputs.numpy_input_fn(
                x=get_input_x(train_data), y=get_input_y(train_data),
                batch_size=train_data.shape[0], shuffle=False)))
        print('[%d] Evaluation metrics:' % i, model.evaluate(
            tf.estimator.inputs.numpy_input_fn(
                x=get_input_x(eval_data), y=get_input_y(eval_data),
                batch_size=eval_data.shape[0], shuffle=False)
        ))

        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=get_input_x(test_data, has_weight=False),
            batch_size=test_data.shape[0], shuffle=False)
        probas = []
        for _, proba in model.predict(pred_input_fn):
            probas.append(proba)
        dump(i, test_data['id'], probas)


def get_input_x(dataframe, has_weight=True):
    raw_features = [col for col in dataframe.columns if col.startswith('feature')]
    x = {
        'raw': np.array(dataframe[raw_features], dtype=np.float32),
        'group': np.array(dataframe['group'], dtype=np.int32)
    }
    if has_weight:
        x['weight'] = np.array(dataframe['weight'], dtype=np.float32)
    return x


def get_input_y(dataframe):
    return np.array(dataframe['label'], dtype=np.int32)


if __name__ == '__main__':
    tf.app.run()
