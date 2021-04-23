
import tensorflow.compat.v1 as tf
import numpy as np
import os
from sklearn.metrics import confusion_matrix

#os.environ['KMP_DUPLICATE_LIB_OK']='True'
# This class might be broken into 2 parts if using GAN later on
class TextCNN():
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters):
        # Plceholders to be filled with input feed_dictionary at training
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x") # To be embedded
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # Start building the network here:
            # Initialize W as the word embedding matrix
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            # The above has output shape: [none, sequence_length, embedding_size]
            # However, tf.nn.con2d() only takes input of rank 4 or higher: [none, height, width, channel]
            # Thus need to add a "channel" dimension by:
            self.embedded_char_expanded = tf.expand_dims(self.embedded_chars, -1) 
            # "-1" adds to the innermost dimension
            pooled_outputs = [] # As per the paper, the pooling layer takes the max of each filter's featuremaps
            # NOTE: We are using multiple filter sizes as per the paper's specs
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-filter_size-"+str(filter_size)):
                    # Define W as the filter matrix (NOTE: different namescope from the W above)
                    # Initialized with truncated normal parameters
                    # The W filter has shape: [height, width, input_channels, output_channels]
                    W = tf.Variable(tf.truncated_normal([filter_size, embedding_size, 1, num_filters],
                                                       stddev=0.1))
                    # Conv layer: valid padding yields output of shape:
                    # [none, sequence_length - filter_size + 1, 1, num_filters]
                    # for dimensions: [none, height, width, channel]
                    # TF document: "(conv2d) has the same type as input and the same outer batch shape."
                    conv = tf.nn.conv2d(self.embedded_char_expanded, W, strides=[1, 1, 1, 1], 
                                       padding="VALID", name="conv")
                    # Biase vector: 1d vector with length=number of output channels of conv
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    # Relu
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                    # TF document: "ksize: The size of the window for each dimension of the input tensor."
                    pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                           strides=[1, 1, 1, 1], padding="VALID", name="pool")
                    # The output now has size: [none, 1, 1, num_filters]
                    pooled_outputs.append(pooled)
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3) # Concatenate on the forth dimension
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
            # The output now has shape: [none, num_filters_total]
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        with tf.name_scope("output"):
            # Fully connected layer
            # Matrix multiplication: (none, num_filters_total)x(num_filters_total, num_classes) = (none, num_classes)    

         #   sess.run(tf.global_variables_initializer())
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            # NOTE: b has dimension of the channels (in this case, num_classes)
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            # Take max of output logits as the predicted class
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            # Mean
            self.loss = tf.reduce_mean(losses)
        with tf.name_scope("accuracy"):
            # NOTE: axis=1 is the row direction, smaller axis number is the outermost dimension
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
            
def optimizer(loss, learning_rate):
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss)
    descend_op = optimizer.apply_gradients(gradients, global_step=global_step) # Apply gradient descend
    return descend_op

def shuffle_data(data, labels):#all arrays here are row vectors? labels and data are columnwise
    np.random.seed(123) # Debugging
    indices = np.arange(labels.shape[0]) #index sequence whose length = len(labels)
    np.random.shuffle(indices) #In place
    shuffled_indices = indices #Useless assignment for clarity- shuffle is in place
    return data[shuffled_indices], labels[shuffled_indices]
    


# Get input_x and input_y batches
def get_train_batch(x_train, y_train, batch_size):
    x_train_shuffle, y_train_shuffle = shuffle_data(x_train, y_train)
    counter = 0
    while counter < len(y_train):
        return_x_train = x_train_shuffle[counter:counter+batch_size]
        return_y_train = y_train_shuffle[counter:counter+batch_size]
        counter = counter + batch_size
        yield return_x_train, return_y_train
        
def get_test_batch(x_test, y_test, batch_size):
    counter = 0
    while counter < len(y_train):
        return_x_train = x_train_shuffle[counter:counter+batch_size]
        return_y_train = y_train_shuffle[counter:counter+batch_size]
        counter = counter + batch_size
        yield return_x_train, return_y_train

def get_batch(XTest, yTest, batchSize):
    #a generator that slices and returns batchSize of XTrain and yTrain instances from top down
    counter = 0;
    while True:
        if counter >= len(yTest):
            break;
        returnXTest = XTest[counter:counter + batchSize];
        returnYTest = yTest[counter:counter + batchSize];
        counter = counter + batchSize;
        yield returnXTest, returnYTest
        
def train_log(log_path, task_name, epoch, loss, train_acc, val_acc):
    #if not os.path.exists(log_path+task_name+'.csv'):
        #os.makedirs(log_path+task_name+'.csv')
    if epoch == 0:
        mode = 'w' # If starting a new training session, overwrite file
    else:
        mode = 'a' # Otherwise append to file
    with open(log_path+task_name+'.csv', mode) as f:
        if mode == 'w': # If overwrite
            header = 'epoch, train_loss, train_acc, val_acc\n'
            f.write(header)
        line = '%d, %f, %f, %f\n' %(epoch, loss, train_acc, val_acc)
        f.write(line)
      
    
# Save model based on TPR and TNR criteria
def save_model_recalls(folder_path, sess, epoch, recall0, recall2):
    if not os.path.exists(folder_path+'/'+task_name):
        os.makedirs(folder_path+'/'+task_name)
   # if recall0 > 0.55 and recall2 > 0.55:
    if recall2 > 0.25:
        saver = tf.train.Saver(max_to_keep = 1)
        saver.save(sess, folder_path+'/'+task_name+'/'+'recall0_'+str(recall0)+'recall2_'+str(recall2)+'_epoch_'+str(epoch)+'_SSL_GAN.ckpt')
    return
        
def train(x_train, y_train, x_test, y_test, dropout_keep_prob, batch_size, num_epochs):
    tf.disable_eager_execution()
    # You can use the same graph in multiple sessions, but not multiple graphs in one session.
    #with tf.Graph().as_default():
    #session_conf = tf.ConfigProto(allow_soft_placement=True, # Fall back on device if not existent
                                  #log_device_placement=True) # Log on device selected
    tf.reset_default_graph()
    # Upper-case variables are global
    print("filter is ")
    print(FILTER_SIZES)
    cnn = TextCNN(sequence_length=x_train.shape[1], num_classes=NUM_CLASSES, vocab_size=VOCABULARY_SIZE,
                 embedding_size=EMBEDDING_SIZE, filter_sizes=FILTER_SIZES, num_filters = NUM_FILTERS)
    descend_op = optimizer(cnn.loss, 5e-4) # Continue the graph structure by adding optimizer
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        preds_return = [] # list of predictions for raw test set at each epoch
        for epoch in range(num_epochs):
            for x_train_batch, y_train_batch in get_batch(x_train, y_train, batch_size):
                #print(y_train_batch)
                feed_dict = { # Key names are python variable names
                    cnn.input_x: x_train_batch,
                    cnn.input_y: y_train_batch,
                    cnn.dropout_keep_prob: dropout_keep_prob
                }
                _, train_loss, train_acc = sess.run(fetches=[descend_op, cnn.loss, cnn.accuracy], feed_dict=feed_dict)

            correct_predictions_concat = []
            predictions_concat = []
            # Validate at the end of each epoch
            for x_test_batch, y_test_batch in get_batch(x_test, y_test, batch_size):
                #print('batch')
                val_feed_dict = {
                    cnn.input_x: x_test_batch,
                    cnn.input_y: y_test_batch,
                    cnn.dropout_keep_prob: 1.0 # No dropout
                }
                correct_predictions, preds = sess.run(fetches=[cnn.correct_predictions, cnn.predictions], 
                                               feed_dict=val_feed_dict)
                #print(correct_predictions.shape)
                correct_predictions_concat = np.concatenate((correct_predictions_concat, correct_predictions))
                predictions_concat = np.concatenate((predictions_concat, preds))
            
            #print(predictions_concat)
            cm = confusion_matrix(np.argmax(y_test, axis=1), predictions_concat)
            print(cm)
            val_acc = np.mean(correct_predictions_concat)
            print('val_acc '+str(val_acc))
            
            save_model_recalls(folder_path, sess, epoch, cm[0][0], cm[2][2])
            # Log information, where train_loss and train_acc are of the last training batch
           # train_log(LOG_PATH, TASK_NAME, epoch, train_loss, train_acc, val_acc)
    return preds_return

def train_cnn(): 
	# Global variables
    global NUM_CLASSES 
    global VOCABULARY_SIZE
    global EMBEDDING_SIZE 
    global FILTER_SIZES 
    global NUM_FILTERS 
    global LOG_PATH 
    global TASK_NAME
    global x_train
    global y_train
    global x_test
    global y_test
    global folder_path
    global task_name 
    global raw_x_test
    global raw_y_test
    global y_test_dummy
    
    
    
    
    

    NUM_CLASSES = 4
    VOCABULARY_SIZE = 1003
    EMBEDDING_SIZE = 80
    FILTER_SIZES = [3, 4, 5]
    NUM_FILTERS = 3
    LOG_PATH = 'training_log/'
    TASK_NAME = 'CNN_base_80_embed'
    x_train = np.load('numpyData/x_train.npy', allow_pickle=True) # Three class
    y_train = np.load('numpyData/y_train.npy') # Three class
    x_test = np.load('numpyData/x_test.npy', allow_pickle=True) # Three class
    y_test = np.load('numpyData/y_test.npy') # Three class
    folder_path = ('savedModels/CNN')
    task_name = "CNN_"+str(EMBEDDING_SIZE)

    raw_x_test = np.load('numpyData/raw_x_test.npy', allow_pickle=True) # Load raw data with 4 classes instead of 3
    raw_y_test = np.load('numpyData/raw_y_test.npy') # Load raw data with 4 classes instead of 3
    y_test_dummy = np.zeros((260, 4))

    preds = train(x_train, y_train, x_test, y_test, 0.75, 60, 5)