import numpy as np
import tensorflow._api.v2.compat.v1 as tf
import sklearn.preprocessing as prep
import pandas as pd

tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

class Autoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.2):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.noisex = self.x+scale*tf.random_normal((n_input,))

        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),
                                                      self.weights['w1']), self.weights['b1']))


        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)


    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights


    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X,
                                                                          self.scale: self.training_scale})
        return cost

    def before_loss(self, X):
        cost = self.sess.run((self.cost), feed_dict={self.x: X,
                                                     self.scale: self.training_scale})
        return cost


    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])

        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})


    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBias(self):
        return self.sess.run(self.weights['b1'])


def standard_scale(X_train):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    return X_train

def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: (start_index + batch_size)]

def DAE(x_train,input_size,training_epochs,batch_size,display_step,lowsize,hidden_size):
    sdne = []
    ###initialize
    for i in range(len(hidden_size)):
        ae = Autoencoder(
            n_input=input_size,
            n_hidden=lowsize,
            transfer_function=tf.nn.softplus,
            optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
            scale=0.2)
        sdne.append(ae)
    Hidden_feature = []
    for j in range(len(hidden_size)):
        if j == 0:
            X_train = standard_scale(x_train)
        else:
            X_train_pre = X_train
            X_train = sdne[j - 1].transform(X_train_pre)
            Hidden_feature.append(X_train)

        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(X_train.shape[0] / batch_size)

            for batch in range(total_batch):
                batch_xs = get_random_block_from_data(X_train, batch_size)

                cost = sdne[j].partial_fit(batch_xs)

                avg_cost += cost / X_train.shape[0] * batch_size
            if epoch % display_step == 0:
                print("Epoch:", "%4d" % (epoch + 1), "cost:", "{:.9f}".format(avg_cost))

        if j == 0:
            feat0 = sdne[0].transform(standard_scale(x_train))
            data1 = pd.DataFrame(feat0)
            print(data1.shape)
            np.set_printoptions(suppress=True)
    return data1

