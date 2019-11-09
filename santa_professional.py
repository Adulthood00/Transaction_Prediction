import tensorflow as tf
import numpy as np
import pandas as pd
import os


path="~/Downloads/santander-customer-transaction-prediction/train.csv"


data=pd.read_csv(path)
data=data.drop("ID_code",axis=1)
predictors = data.drop("target", axis=1) 
target = data["target"]

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(predictors)
predictors=scaler.transform(predictors)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( predictors, target, test_size = 0.01)

y_train = y_train.values
X_valid=X_test
y_valid=y_test.values

m, n = X_train.shape


n_hidden1 = 1000
n_hidden2 = 1000
n_hidden3 = 500
n_outputs = 2

X=tf.placeholder(tf.float32, shape=(None,n) ,name="X")
y=tf.placeholder(tf.int32, shape=(None),name="y")

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

with tf.name_scope("santa_dnn"):
    reg=tf.contrib.layers.l1_l2_regularizer(0.001,0.001)
    hidden1=tf.layers.dense(X,n_hidden1,activation=tf.nn.tanh,kernel_regularizer=reg)
    dropout1=tf.layers.dropout(hidden1,rate=0.3)
    hidden2=tf.layers.dense(dropout1,n_hidden2, activation=tf.nn.tanh,kernel_regularizer=reg)
    dropout2=tf.layers.dropout(hidden2,rate=0.3)
    hidden3=tf.layers.dense(dropout2,n_hidden3, activation=tf.nn.tanh,kernel_regularizer=reg)
    dropout3=tf.layers.dropout(hidden3,rate=0.3)
    logits=tf.layers.dense(dropout3,n_outputs,name="logits")        
    y_proba = tf.nn.softmax(logits,name="y_proba")

with tf.name_scope("santa_loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_proba)
    loss = tf.reduce_mean(xentropy)
    loss_summary = tf.summary.scalar('log_loss', loss)

learning_rate=0.001

with tf.name_scope("santa_train"):
    #optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op=optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(y_proba, y, 1,name="correct")
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32),name="accuracy")    
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

init=tf.global_variables_initializer()
saver=tf.train.Saver()

    
from datetime import datetime

def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

logdir = log_dir("mnist_dnn")
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 20000
batch_size = 250
n_batches = int(np.ceil(m / batch_size))

checkpoint_path = "/tmp/my_deep_santa_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "./my_deep_santa_model"

best_loss = np.infty
epochs_without_progress = 0
max_epochs_without_progress = 100000

with tf.Session() as sess:
    if os.path.isfile(checkpoint_epoch_path):
        # if the checkpoint file exists, restore the model and load the epoch number
        with open(checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        print("Training was interrupted. Continuing at epoch", start_epoch)
        saver.restore(sess, checkpoint_path)
    else:
        start_epoch = 0
        sess.run(init)

    for epoch in range(start_epoch, n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val, loss_val, accuracy_summary_str, loss_summary_str = sess.run([accuracy, loss, accuracy_summary, loss_summary], feed_dict={X: X_valid, y: y_valid})
        file_writer.add_summary(accuracy_summary_str, epoch)
        file_writer.add_summary(loss_summary_str, epoch)
        if epoch % 100 == 0:
            print("Epoch:", epoch,
                  "\tValidation accuracy: {:.3f}%".format(accuracy_val * 100),
                  "\tLoss: {:.5f}".format(loss_val))
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, "wb") as f:
                f.write(b"%d" % (epoch + 1))
            if loss_val < best_loss:
                saver.save(sess, final_model_path)
                best_loss = loss_val
            else:
                epochs_without_progress += 100
                if epochs_without_progress > max_epochs_without_progress:
                    print("Early stopping")
                    break
