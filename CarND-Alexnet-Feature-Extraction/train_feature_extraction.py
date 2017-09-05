import time
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet


# TODO: Load traffic signs data.
nb_classes = 43
epochs = 1
batch_size = 128
traffic_sign_data_file = './train.p'

with open(traffic_sign_data_file, mode = 'rb') as f:
    traffic_sign_data = pickle.load(f)

num_of_samples = 500
data_features = traffic_sign_data['features'][:num_of_samples]
data_labels = traffic_sign_data['labels'][:num_of_samples]

print(data_features.shape)
print(data_labels.shape)
# TODO: Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(data_features, data_labels, test_size = 0.2, random_state = 0)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int64, (None))
resized = tf.image.resize_images(x, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape = shape, stddev = 0.01, dtype = tf.float32))
fc8b = tf.Variable(tf.zeros(shape = nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0005)
training_operation = optimizer.minimize(loss_operation, var_list=[fc8W, fc8b])
init_operation = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(logits, 1), y)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        start = offset
        end = offset + batch_size
        batch_x, batch_y = X_data[start:end], y_data[start:end]
        loss, accuracy = sess.run([loss_operation, accuracy_operation], feed_dict = {x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))
        return total_loss/num_examples, total_accuracy/num_examples

# TODO: Train and evaluate the feature extraction model.
with tf.Session() as sess:
    sess.run(init_operation)
    num_examples = len(X_train)
    
    for i in range(epochs):
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, num_examples, batch_size):
            start = offset
            end = offset + batch_size
            batch_x, batch_y = X_train[start:end], y_train[start:end]
            sess.run(training_operation, feed_dict = {x: batch_x, y: batch_y})
            
        validation_loss, validation_accuracy = evaluate(X_valid, y_valid)
        training_loss, training_accuracy = evaluate(X_train, y_train)
        print("EPOCH {} ...".format(i+1))
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print() 
    