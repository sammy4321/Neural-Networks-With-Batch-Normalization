import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation
import tensorflow as tf
import matplotlib.pyplot as plt

features = pd.read_csv("dataset/cancer_attributes.txt")
#features=preprocessing.scale(features)
targets = pd.read_csv("dataset/cancer_classes.txt")

x_train,x_test,y_train,y_test = cross_validation.train_test_split(features,targets,test_size=0.2)
#x_train=preprocessing.scale(x_train)
#x_test=preprocessing.scale(x_test)

training =0
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
#batch_size = 100

x = tf.placeholder('float', [None, 9])
y = tf.placeholder('float')

def neural_network_model(data):
	data = tf.contrib.layers.batch_norm(data,center=True, scale=True,is_training=True)
	l1=tf.contrib.layers.fully_connected(data,num_outputs=n_nodes_hl1,activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float32),biases_initializer=tf.zeros_initializer(),trainable=True)
	#l1=tf.contrib.layers.batch_norm(l1,center=True, scale=True,is_training=True)
	l2=tf.contrib.layers.fully_connected(l1,num_outputs=n_nodes_hl2,activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float32),biases_initializer=tf.zeros_initializer(),trainable=True)
	#l2=tf.contrib.layers.batch_norm(l2,center=True, scale=True,is_training=True)
	l3=tf.contrib.layers.fully_connected(l2,num_outputs=n_nodes_hl3,activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float32),biases_initializer=tf.zeros_initializer(),trainable=True)
	#l3=tf.contrib.layers.batch_norm(l3,center=True, scale=True,is_training=True)
	output=tf.contrib.layers.fully_connected(l3,num_outputs=n_classes,activation_fn=None,weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float32),biases_initializer=tf.zeros_initializer(),trainable=True)
	return output
def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    hm_epochs = 100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        cost_list=[]
        accuracy_list=[]
        no_epochs_list=[]
        for epoch in range(hm_epochs):
            epoch_loss = 0
            _, c = sess.run([optimizer, cost], feed_dict={x: x_train, y: y_train})
            epoch_loss += c
            cost_list.append(epoch_loss)
            no_epochs_list.append(epoch)
            acc=accuracy.eval({x:x_test, y:y_test})
            acc=acc*100
            accuracy_list.append(acc)
            print('Iteration : ',epoch+1,'Accuracy : ',acc)
    
    plt.subplot(1,3,1)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Accuracy of test data")
    plt.plot(no_epochs_list,accuracy_list,color='b')

    plt.subplot(1,3,3)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.plot(no_epochs_list,cost_list,color='r')
    plt.show()

        
        

train_neural_network(x)