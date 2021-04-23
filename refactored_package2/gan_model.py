import numpy as np
import time
import os

import tensorflow.compat.v1 as tf
#from tensorflow.examples.tutorials.mnist import input_data

from matplotlib import pyplot as plt
import math
#%matplotlib inline
import datetime
from sklearn.metrics import confusion_matrix
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
tf.__version__


# NLP related:
vocab_size = 1003
embedding_size = 80
sequence_length = 64
filter_sizes = [3,4,5]
num_filters = 3
# GAN related:
task_name = "NLP_embedding_"+str(embedding_size)

npSeed = 123
np.random.seed(npSeed)
x_height, x_width = [sequence_length, embedding_size]
num_channels = 1
num_classes = 4 #Undamaged, cracked/spalling
latent_size = 100
labeled_rate = 1.0 #1.0, 0.5, 0.1 this limits the knowledge base of the learning
#unlabeled_supp_rate = 0 # percentage of unlabeled data to be supplemented to the learning
c_ul = 0
# task_path = 'numpyData/BINARY_CR_16_1_split_128_128'
# When you wake up, UNCOMMENT THIS line below****
task_path = 'numpyData/'
# All data (labeled and unlabeled) by class
# For NLP: the data input is [[sentence1], [sentence2], ...] where [sentencei] is an array of floats
train_data_by_class, test_data_by_class, train_label_by_class, test_label_by_class = [], [], [], []
#train_mask_by_class = []
#unlabeled_indices_by_class = []
labeled_indices_by_class = [] # Marked for labeled data selection to batch

def loadClasses(): 
    global train_data_by_class
    global train_label_by_class
    global test_data_by_class
    global test_label_by_class
    
    train_data_by_class = np.load('numpyData/x_train_by_class.npy', allow_pickle=True)
    train_label_by_class = np.load('numpyData/y_train_by_class.npy', allow_pickle=True)
    test_data_by_class = np.load('numpyData/x_test_by_class.npy', allow_pickle=True)
    test_label_by_class = np.load('numpyData/y_test_by_class.npy', allow_pickle=True)







unlabeled_data = np.zeros((0, x_width, x_height, num_channels)) # In order to randomly select from
# all unlabeled data without considering classes
labeled_data_baseline = np.zeros((0, x_width, x_height, num_channels)) # In order to export 0.25 labeled 
# Load real data:
# for i in range(num_classes):
#     train_data_by_class.append(np.load(task_path+'/class_'+str(i)+'/trainX.npy'))
#     train_label_by_class.append(np.load(task_path+'/class_'+str(i)+'/trainy.npy'))
                        
#     test_data_by_class.append(np.load(task_path+'/class_'+str(i)+'/testX.npy'))
#     test_label_by_class.append(np.load(task_path+'/class_'+str(i)+'/testy.npy'))
#     print("clas+ "+str(i)+'has '+str(len(train_label_by_class[i])))










def normalize(x):
    # normalize data
#     x /= 255.0
    x = (x - 127.5) / 127.5
    return x.reshape((-1, x_height, x_width, 3)) #x is 4 dimensional-- (num_images, height, width, depth)

#save masked/labeled image to numpy array
def save_masked(toFolder):
    for i in range(num_classes):
        if not os.path.exists('numpyData/'+toFolder+'/class_'+str(i)):
            os.makedirs('numpyData/'+toFolder+'/class_'+str(i))
        np.save('numpyData/'+toFolder+'/class_'+str(i)+'/trainX', train_data_by_class[i][train_mask_by_class[i].astype(bool)])
        np.save('numpyData/'+toFolder+'/class_'+str(i)+'/trainy', train_label_by_class[i][train_mask_by_class[i].astype(bool)])

#save_masked("labeled_for_cnn_seed_"+str(npSeed)+'_rate_'+str(labeled_rate))

#shuffle data array, labels array, and labeledMask array, each image's properties should remain consistant (labeled/unlabeled)
#during the whole experiment. Instead of assigning labeled mask per next batch, it should be globally defined
#prior to running the experiment --BZ, August, 2019
def shuffle_data(data, labels, labeledMask):#all arrays here are row vectors? labels and data are columnwise
    #np.random.seed(123)#for debugging purpose
    indices = np.arange(labels.shape[0]) #index sequence whose length = len(labels)
    np.random.shuffle(indices) #In place
    shuffled_indices = indices #Useless assignment for clarity- shuffle is in place
    if labeledMask is None:#for cases used by get_test_batch() function--BZ, August, 2019
        return data[shuffled_indices], labels[shuffled_indices]
    else:
        return data[shuffled_indices], labels[shuffled_indices], labeledMask[shuffled_indices]

#new batch functions:--BZ, August, 2019
#because labeled mask should stay with the same images after each shuffle (which happens at the start of each new epoch),
#get_batch() and get_labeled_mask() should be merged into one function

#new batch functions:--BZ, August, 2019
#because labeled mask should stay with the same images after each shuffle (which happens at the start of each new epoch),
#get_batch() and get_labeled_mask() should be merged into one function
def get_training_batch_and_labeled_mask(XTrain, yTrain, labeledMask, batchSize):
    #first shuffle the indices:
    XTrainRandom, yTrainRandom, labeledMaskRandom = shuffle_data(XTrain, yTrain, labeledMask);
    #a generator that slices and returns batchSize of XTrain and yTrain instances from top down
    counter = 0;
    while True:
        if counter >= len(yTrain):
            break;
        returnXTrain = XTrainRandom[counter:counter + batchSize];
        returnYTrain = yTrainRandom[counter:counter + batchSize];
        returnLabeledMask = labeledMaskRandom[counter:counter + batchSize];
        counter = counter + batchSize;
        yield returnXTrain, returnYTrain, returnLabeledMask
def get_test_batch(XTest, yTest, batchSize): #essentially the same function as above, restated here for explicity
    #first shuffle the indices:
    XTestRandom, yTestRandom = shuffle_data(XTest, yTest, None);
    #a generator that slices and returns batchSize of XTrain and yTrain instances from top down
    counter = 0;
    while True:
        if counter >= len(yTest):
            break;
        returnXTest = XTestRandom[counter:counter + batchSize];
        returnYTest = yTestRandom[counter:counter + batchSize];
        counter = counter + batchSize;
        yield returnXTest, returnYTest

def get_train_batch(XTrain, yTrain, batchSize):
    get_test_batch(XTrain, yTrain, batchSize)
        
def get_balance_train_batch(train_data_by_class, train_label_by_class, train_mask_by_class, batchSize):
    numEachClass = int(np.floor(batchSize / num_classes))
    returnData = np.zeros((0, x_width, x_height, num_channels))
    returnLabel = np.zeros((0, num_classes))
    returnLabeledMask = np.zeros((0))
    for i in range(num_classes):
        train_data = train_data_by_class[i]
        train_label = train_label_by_class[i]
        train_mask = train_mask_by_class[i]
        indices = [v for v in range(len(train_label))]
        np.random.shuffle(indices)
        selectIndices = indices[0:numEachClass]
        returnData = np.concatenate((returnData, train_data[selectIndices]))
        returnLabel = np.concatenate((returnLabel, train_label[selectIndices]))
        returnLabeledMask = np.concatenate((returnLabeledMask, train_mask[selectIndices]))
    return returnData, returnLabel, returnLabeledMask

def get_balance_train_batch_2(train_data_by_class, train_label_by_class, train_mask_by_class, batchSize):
    numEachClass = int(batchSize / num_classes)
    numEachClass_unlabeled_labeled = [numEachClass - int(numEachClass*labeled_rate), int(numEachClass*labeled_rate)]
#     print(numEachClass)
#     print(numEachClass_unlabeled_labeled)
    returnData = np.zeros((0, x_width, x_height, num_channels))
    returnLabel = np.zeros((0, num_classes))
    returnLabeledMask = np.zeros((0))
    for i in range(num_classes):
        for j in [0,1]: #for unlabeled and labeled
            indices = []
            for k in range(len(train_mask_by_class[i])):
                if train_mask_by_class[i][k] == j:
                    indices.append(k)
#             print('label status_'+str(j))
#             print(indices)
            np.random.shuffle(indices)
            selectIndices = indices[0:numEachClass_unlabeled_labeled[j]]
#             print('selected')
#             print(selectIndices)
            train_data = train_data_by_class[i]
            train_label = train_label_by_class[i]
            train_mask = train_mask_by_class[i]
            
            returnData = np.concatenate((returnData, train_data[selectIndices]))
            returnLabel = np.concatenate((returnLabel, train_label[selectIndices]))
            returnLabeledMask = np.concatenate((returnLabeledMask, train_mask[selectIndices]))
    return returnData, returnLabel, returnLabeledMask

# Input: c_ul is the relative portion ratio related to each class in a batch
def get_balance_train_batch_3(train_data_by_class, train_label_by_class, batchSize, c_ul):
    numEachPortion = int(batchSize/(num_classes+1+c_ul)) # One portion of fake,c_ul portion of unlabeled
    returnData = np.zeros((0, sequence_length), dtype='int')
    returnLabel = np.zeros((0, num_classes))
    returnLabeledMask = np.zeros((0))
    # First load labeled Data (mask is 1)
    for i in range(num_classes):
        labeled_is = np.random.choice(labeled_indices_by_class[i], numEachPortion)
        returnData = np.concatenate((returnData, train_data_by_class[i][labeled_is]))
        returnLabel = np.concatenate((returnLabel, train_label_by_class[i][labeled_is]))
        returnLabeledMask = np.concatenate((returnLabeledMask, np.ones(numEachPortion))) # 1 for labeled
    # Then load unlabeled Data (mask is 0)
    if len(unlabeled_data) is not 0:
        num_unlabeled = numEachPortion * c_ul
        unlabeled_is = np.random.choice([i for i in range(len(unlabeled_data))], num_unlabeled)
        returnData = np.concatenate((returnData, unlabeled_data[unlabeled_is]))
        returnLabel = np.concatenate((returnLabel, np.zeros((num_unlabeled, num_classes))))
        returnLabeledMask = np.concatenate((returnLabeledMask, np.zeros(num_unlabeled))) # 0 for unlabeled
    return returnData, returnLabel, returnLabeledMask

def get_test_batch(test_data_by_class, test_label_by_class, batchSize):
    XTest = np.zeros((0, sequence_length), dtype='int')
    yTest = np.zeros((0, num_classes))
    for i in range(num_classes):
        XTest = np.concatenate((XTest, test_data_by_class[i]))
        yTest = np.concatenate((yTest, test_label_by_class[i]))
    #first shuffle the indices:
    XTestRandom, yTestRandom = shuffle_data(XTest, yTest, None);
    #a generator that slices and returns batchSize of XTrain and yTrain instances from top down
    counter = 0;
    while True:
        if counter >= len(yTest):
            break;
        returnXTest = XTestRandom[counter:counter + batchSize];
        returnYTest = yTestRandom[counter:counter + batchSize];
        counter = counter + batchSize;
        yield returnXTest, returnYTest









def D(data_source, x_real, x_fake, dropout_rate, is_training, reuse = True, print_summary = True):
    # data_source is a string, either "fake" or "real", which determines whether do to the word
    # embedding lookup to avoid non-differentiability issues.
    # discriminator (x -> n + 1 class)

    with tf.variable_scope('Discriminator', reuse = reuse) as scope:
        # Embedding layer
        # Input x has shape [batch_size, 63] where 63 is the sequence length
        W_embed = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W_embed")
        embedded_chars = tf.nn.embedding_lookup(W_embed, x_real)
        # Add a channel dimension:
        embedded_char_expanded = tf.expand_dims(embedded_chars, -1) 
        # Output size: [batch_size, sequence_length, embedding_size, 1]
        
        print('fake shape is!')
        print(x_fake.get_shape())
        print('embed_char_expand shape is!')
        print(embedded_char_expanded.get_shape())
        # conditional pipeline!
        def f1(): return embedded_char_expanded
        def f2(): return x_fake
        real_or_fake = tf.math.equal('real', data_source)
        input_x = tf.cond(real_or_fake, f1, f2)
        
        print('input_x shape is!') # [batch, seq_len, embed_size, 1]
        print(input_x.get_shape())
        
        pooled_outputs = [] # As per the paper, the pooling layer takes the max of each filter's featuremaps
        # NOTE: We are using multiple filter sizes as per the paper's specs
        for i, filter_size in enumerate(filter_sizes):
            #with tf.name_scope("conv-maxpool-filter_size-"+str(filter_size)):
            # Define W as the filter matrix (NOTE: different namescope from the W above)
            # Initialized with truncated normal parameters
            # The W filter has shape: [height, width, input_channels, output_channels]
            W = tf.Variable(tf.truncated_normal([filter_size, embedding_size, 1, num_filters],
                                               stddev=0.1))
            # Conv layer: valid padding yields output of shape:
            # [none, sequence_length - filter_size + 1, 1, num_filters]
            # for dimensions: [none, height, width, channel]
            # TF document: "(conv2d) has the same type as input and the same outer batch shape."
            conv = tf.nn.conv2d(input_x, W, strides=[1, 1, 1, 1], 
                               padding="VALID", name="conv")
            # Biase vector: 1d vector with length=number of output channels of conv
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            # Relu
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
            lrelu3 = tf.maximum(0.2 * h, h)
            # TF document: "ksize: The size of the window for each dimension of the input tensor."
            pooled = tf.nn.max_pool(lrelu3, ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                   strides=[1, 1, 1, 1], padding="VALID", name="pool")
            # The output now has size: [none, 1, 1, num_filters]
            pooled_outputs.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3) # Concatenate on the forth dimension
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        
        
        # The output now has shape: [none, num_filters_total]
        #with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, rate=dropout_rate)
        #with tf.name_scope("output"):
        # Fully connected layer
        # Matrix multiplication: (none, num_filters_total)x(num_filters_total, num_classes) = (none, num_classes)
        W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes+1], stddev=0.1), name="W")
        # NOTE: b has dimension of the channels (in this case, num_classes)
        b = tf.Variable(tf.constant(0.1, shape=[num_classes+1]), name="b")
        fc = tf.nn.xw_plus_b(h_drop, W, b, name="scores") #Logits
        output = tf.nn.softmax(fc)
        return h_pool_flat, fc, output, real_or_fake
def G(z, is_training, reuse = False, print_summary = False):
    # generator (z -> x)

    with tf.variable_scope('Generator', reuse = reuse) as scope:
        #z is 100*1
        fc1 = tf.layers.dense(z, 16*int(embedding_size/4)*128)
        # layer 0
        z_reshaped = tf.reshape(fc1, [-1, 16, int(embedding_size/4), 128])

    
        
        # layer 1
        deconv1 = tf.layers.conv2d_transpose(z_reshaped,
                                             filters = 128,
                                             kernel_size = [3, 3],
                                             strides = [2, 2],
                                             padding = 'same')
        batch_norm1 = tf.layers.batch_normalization(deconv1, training = is_training, momentum=0.8)
        relu1 = tf.nn.relu(batch_norm1)
        #64*64*64
        # layer 2
        deconv2 = tf.layers.conv2d_transpose(relu1,
                                             filters = 64,
                                             kernel_size = [3, 3],
                                             strides = [2, 2],
                                             padding = 'same')
        batch_norm2 = tf.layers.batch_normalization(deconv2, training = is_training, momentum=0.8)
        relu2 = tf.nn.relu(batch_norm2)
        #128*128*3
        # layer 3
        #deconv3 = tf.layers.conv2d_transpose(relu2,
                                             #filters = 64,
                                             #kernel_size = [3, 3],
                                             #strides = [1, 1],
                                             #padding = 'same')
        #batch_norm3 = tf.layers.batch_normalization(deconv3, training = is_training)
        #relu3 = tf.nn.relu(batch_norm3)

        # layer 4 - do not use Batch Normalization on the last layer of Generator
        deconv4 = tf.layers.conv2d_transpose(relu2,
                                             filters = num_channels,
                                             kernel_size = [3, 3],
                                             strides = [1, 1],
                                             padding = 'same')
        output = deconv4
        #tanh4 = tf.tanh(deconv4, name="G_output")
        #print('tanh shape')
        #print(tanh4.get_shape())
        assert output.get_shape()[1:] == [x_height, x_width, num_channels]
        
#         decoded_sentences_list = [] # A list of sentences [batch_size, seq_length]
#         for d in deconv4: # For each image in batch
#             decoded_word_list = []
#             for word in d: # For each sentence in sentence image
#                 word_reduce = tf.squeeze(word, axis=1) # take out the channel dimension
#                 # Iterate each row of W_embed and find the row with the closest vector distance
#                 smallest_norm = np.inf
#                 smallest_norm_id = None
#                 for r in len(W_embed):
#                     norm = tf.norm(tanh4[d], word_reduce)
#                     if norm <= smallest_norm:
#                         smallest_norm = norm
#                         smallest_norm_id = r
#                 decoded_word_id = smallest_norm_id # To avoid confusion
#                 decoded_word_list.append(decoded_word_id)
#             decoded_sentences_list.append(decoded_word_list)
#         assert len(decoded_sentences_list[0]) == seq_length
        

        if print_summary:
            print('Generator summary:\n z: %s\n' \
                  ' G0: %s\n G1: %s\n G2: %s\n G3: %s\n G4: %s\n' %(z.get_shape(),
                                                                    z_reshaped.get_shape(),
                                                                    relu1.get_shape(),
                                                                    relu2.get_shape(),
                                                                    relu3.get_shape(),
                                                                    tanh4.get_shape()))
        return output

#build model for each batch using D() and G() functions
def build_model(x_real, z, label, dropout_rate, is_training, print_summary = False):
    # build model
    #generate fake images
    x_fake = G(z, is_training, reuse = False, print_summary = print_summary)
    
    
    #Discriminator on real data (labeled and unlabeled)  flatten5, fc5, output
    D_real_features, D_real_logit, D_real_prob, rf_real = D(tf.constant('real', dtype=tf.string), x_real, x_fake, dropout_rate, is_training,
                                                   reuse = False, print_summary = print_summary)
    
    #Discriminator for fake images
    D_fake_features, D_fake_logit, D_fake_prob, rf_fake = D(tf.constant('fake', dtype=tf.string), x_real, x_fake, dropout_rate, is_training,
                                                   reuse = True, print_summary = print_summary)

    return D_real_features, D_real_logit, D_real_prob, D_fake_features, D_fake_logit, D_fake_prob, x_fake, rf_real, rf_fake

#model that only contains the discriminator for production model
def build_model_production(x_real, label, dropout_rate, is_training, print_summary = False):
    x, dropout1, batch_norm2, dropout3, dropout4, D_real_features, D_real_logit, D_real_prob, conv1 = D(x_real, dropout_rate, is_training,
                                                   reuse = False, print_summary = print_summary)
    correct_prediction = tf.equal(tf.argmax(D_real_prob[:, :-1], 1),#arg max returns the indices--Bill Zhai Aug, 2019
                                  tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
    return x, dropout1, batch_norm2, dropout3, dropout4, D_real_features, D_real_logit, D_real_prob, accuracy, correct_prediction, conv1

def prepare_labels(label):
    # add extra label for telling apart fake data from real data
    #essentially appending another column to the very end of the matrix which is for the 'fake' class
    #uses one hot, so this newly appended column is all zeros--Bill Zhai Aug, 2019
    extended_label = tf.concat([label, tf.zeros([tf.shape(label)[0], 1])], axis = 1)

    return extended_label
def loss_accuracy(D_real_features, D_real_logit, D_real_prob, D_fake_features,
                  D_fake_logit, D_fake_prob, extended_label, labeled_mask):
    epsilon = 1e-8 # used to avoid NAN loss
    # *** Discriminator loss ***
    # supervised loss
    # which class the real data belongs to
    tmp = tf.nn.softmax_cross_entropy_with_logits(logits = D_real_logit,#cross_entrypy_with_logits is the only function available
                                                  labels = extended_label)
    #question: is this an over simplification?--no, becuase tmp is a vector: num_samples_in_batch * 1
    D_L_supervised = tf.reduce_sum(labeled_mask * tmp) / (tf.reduce_sum(labeled_mask)+1e-6) # to ignore unlabeled data
                                                                                     

    # unsupervised loss
    # data is real
    prob_real_be_real = 1 - D_real_prob[:, -1] + epsilon #"-1" signifies the last column which is probabilities of being "fake"
    tmp_log = tf.log(prob_real_be_real)
    D_L_unsupervised1 = -1 * tf.reduce_mean(tmp_log)

    # data is fake
    prob_fake_be_fake = D_fake_prob[:, -1] + epsilon
    tmp_log = tf.log(prob_fake_be_fake)
    D_L_unsupervised2 = -1 * tf.reduce_mean(tmp_log)

    D_L = D_L_supervised + D_L_unsupervised1 + D_L_unsupervised2

    # *** Generator loss ***
    # fake data is mistaken to be real
    #prob_fake_be_real = 1 - D_fake_prob[:, -1] + epsilon
    #tmp_log =  tf.log(prob_fake_be_real)
    #G_L1 = -1 * tf.reduce_mean(tmp_log)
    G_L1 = 0.0 # Due to non-differentiable decoding

    # Feature Maching
    tmp1 = tf.reduce_mean(D_real_features, axis = 0)
    tmp2 = tf.reduce_mean(D_fake_features, axis = 0)
    G_L2 = tf.reduce_mean(tf.square(tmp1 - tmp2))

    G_L = G_L1 + G_L2

    # accuracy--This is cross validation accuracy within the training set--Nov, 2019
    correct_prediction = tf.equal(tf.argmax(D_real_prob[:, :-1], 1),#arg max returns the indices--Bill Zhai Aug, 2019
                                  tf.argmax(extended_label[:, :-1], 1), name='correct_prediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #cast boolean to float32--Bill Zhai Aug, 2019
    
    
    return D_L_supervised, D_L_unsupervised1, D_L_unsupervised2, D_L, G_L, accuracy, correct_prediction


def optimizer(D_Loss, G_Loss, D_learning_rate, G_learning_rate):
    # D and G optimizer
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        all_vars = tf.trainable_variables()
        D_vars = [var for var in all_vars if var.name.startswith('Discriminator')]
        G_vars = [var for var in all_vars if var.name.startswith('Generator')]
        #print('D_vars:')
        #print(D_vars)
        D_optimizer = tf.train.AdamOptimizer(D_learning_rate).minimize(D_Loss, var_list = D_vars)
        G_optimizer = tf.train.AdamOptimizer(G_learning_rate).minimize(G_Loss, var_list = G_vars)
        return D_optimizer, G_optimizer
    
#Assume len(data) > 5 --BZ Nov 30, 2019

def plot_fake_data(data, epoch):
    # visualize some data generated by G
    data = (1/2.5) * data + 0.5
    fig, axs = plt.subplots(len(data), figsize=(30,30))
    cnt = 0
    for j in range(len(data)):
        #print(j)
        #print(data[cnt, :, :, :])
        axs[j].imshow(data[cnt, :, :, :])
        axs[j].axis("off")
        cnt = cnt + 1
    print('graphed!')        
    if not os.path.exists("./training_fake_figure"):
        os.mkdir("./training_fake_figure");
    plt.savefig("training_fake_figure/%d.jpg" % epoch)
    plt.close()

def save_fake_image(data, epoch):
    #if not os.path.exists("./training_fake_imageMatrix"):
        #os.mkdir("./training_fake_imageMatrix");
    #np.save("./training_fake_imageMatrix/epoch_"+str(epoch), data)
    #only plot the last image of the batch
    if not os.path.exists("./training_fake_figure"):
        os.mkdir("./training_fake_figure")
    plt.imshow(data[-1]/2+0.5)
    plt.set_cmap('hot')
    plt.axis('off')
    #print(data[-1])
    plt.savefig("training_fake_figure/%d.jpg" % epoch)
    plt.close()

def save_fake_sentence(data, epoch, w_embed): # logmode can be 'a' for 'w' (append or overwrite)
    if not os.path.exists("./training_fake_sentence"):
        os.mkdir("./training_fake_figure")
    if epoch == 0:
        logmode = 'w'
    else:
        logmode = 'a'
    
    with open(file_path, logmode) as f: # start overwriting
        if epoch == 0:
            header = 'epoch, train_loss_D, train_loss_G,' \
                         'train_Acc, val_Acc, recall0, recall1, recall2\n'
            f.write(header)
        
        # From numbers back to sentences
        w_embed_transpose = np.transpose(w_embed)
        
            
        
        
        
        tf.nn.embedding_lookup(w_embed, ids, max_norm=None, name=None)
        line = '%d, %f, %f, %f, %f, %f, %f, %f\n' %(epoch, train_loss_D, train_loss_G, train_Acc,
                                                cv_Acc, recall0, recall1, recall2)
        f.write(line)

def save_model_on_improvement(file_path, sess, cv_acc, cv_accs):
  #  # save model when there is improvemnet in cv_acc value
    if cv_accs == [] or cv_acc >= np.max(cv_accs):
        saver = tf.train.Saver(max_to_keep = 1)
        saver.save(sess, file_path)
        print('Model saved')
    print('')

#save the top 5 best model based on validation accuracy
def save_model_top_five(folder_path, sess, cv_acc, cv_accs):
    #cv_acc is inside cv_accs
    if not os.path.exists(folder_path+'/'+task_name):
        os.mkdir(folder_path+'/'+task_name)
    sortedAccs = np.sort(cv_accs)
    for i in range(len(cv_accs)):
        if i >= 5:
            return
        if cv_acc >= sortedAccs[i]:
            saver = tf.train.Saver(max_to_keep = 1)
            saver.save(sess, folder_path+'/'+task_name+'/'+'_top_'+str(i+1)+'_SSL_GAN.ckpt')
            return

# Save model based on TPR and TNR criteria
def save_model_TPR_TNR(folder_path, sess, epoch, cv_acc, cv_accs, TPR, TNR, TPRs):
    if not os.path.exists(folder_path+'/'+task_name):
        os.mkdir(folder_path+'/'+task_name)
    sortedTPRs = np.sort(TPRs)[::-1] # Sort in Descending Order!
    for i in range(len(sortedTPRs)):
        if i >= 5:
            return
        #if TPR >= sortedTPRs[i] and TPR >= 0.88 and TNR >= 0.91:
        if TPR >= sortedTPRs[i]:
            print('save model')
            saver = tf.train.Saver(max_to_keep = 1)
            saver.save(sess, folder_path+'/'+task_name+'/'+'_TPR_top_'+str(i+1)+'_epoch_'+str(epoch)+'_SSL_GAN.ckpt')
            return
        
# Save model based on TPR and TNR criteria
def save_model_recalls(folder_path, sess, epoch, cv_acc, cv_accs, recall0, recall1, recall2):
    if not os.path.exists(folder_path+'/'+task_name):
        os.makedirs(folder_path+'/'+task_name)
  #  print(recall0)

    #if recall0 > 0.55 and recall2 > 0.55:
    if recall2 > 0.1:
        saver = tf.train.Saver(max_to_keep = 1)
        saver.save(sess, folder_path+'/'+task_name+'/'+'recall0_'+str(recall0)+'recall2_'+str(recall2)+'_epoch_'+str(epoch)+'_SSL_GAN.ckpt')
    return

def save_model(file_path, sess):
    saver = tf.train.Saver(max_to_keep = 1)
    saver.save(sess, file_path)
    print('Every 500 model saved')

def log_loss_acc_binary(file_path, epoch, train_loss_D, train_loss_G, train_Acc,
                 cv_Acc, recall0, recall1, recall2, log_mode):
    # log train and cv losses as well as accuracy
    mode = log_mode if epoch == 0 else 'a'

    with open(file_path, mode) as f:
        if mode == 'w':
            header = 'epoch, train_loss_D, train_loss_G,' \
                     'train_Acc, val_Acc, recall0, recall1, recall2\n'
            f.write(header)

        line = '%d, %f, %f, %f, %f, %f, %f, %f\n' %(epoch, train_loss_D, train_loss_G, train_Acc,
                                                cv_Acc, recall0, recall1, recall2)
        f.write(line)
def log_loss_acc_ternary(file_path, epoch, train_loss_D, train_loss_G, train_Acc,
                 cv_Acc, cm00, cm01, cm02, cm10, cm11, cm12, cm20, cm21, cm22, log_mode):
    # log train and cv losses as well as accuracy
    mode = log_mode if epoch == 0 else 'a'

    with open(file_path, mode) as f:
        if mode == 'w':
            header = 'epoch, train_loss_D, train_loss_G,' \
                     'train_Acc, val_Acc, cm00, cm01, cm02, cm10, cm11, cm12, cm20, cm21, cm22\n'
            f.write(header)

        line = '%d, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n' %(epoch, train_loss_D, train_loss_G, train_Acc,
                                                cv_Acc, cm00, cm01, cm02, cm10, cm11, cm12, cm20, cm21, cm22)
        f.write(line)
        
def log_loss_acc_diagonals(file_path, epoch, train_loss_D, train_loss_G, train_Acc,
                 cv_Acc, cm00, cm11, cm22, cm33, log_mode):
    # log train and cv losses as well as accuracy
    mode = log_mode if epoch == 0 else 'a'

    with open(file_path, mode) as f:
        if mode == 'w':
            header = 'epoch, train_loss_D, train_loss_G,' \
                     'train_Acc, val_Acc, cm00, cm11, cm22, cm33\n'
            f.write(header)

        line = '%d, %f, %f, %f, %f, %f, %f, %f, %f\n' %(epoch, train_loss_D, train_loss_G, train_Acc,
                                                cv_Acc, cm00, cm11, cm22, cm33)
        f.write(line)

#Modified by BZ on Nov 3, 2019
#Note: predictions vector has binary entries: 1 corresponds to correct prediction, 0 is wrong prediction
def compute_val_accuracy(correct_predictions):
    return np.sum(correct_predictions)/len(correct_predictions)


def train_SSL_GAN(batch_size, num_epochs, train_data_by_class, train_label_by_class, test_data_by_class, test_label_by_class):
    numTrain = sum([len(c) for c in train_label_by_class])
    numTest = sum([len(c) for c in test_label_by_class])
    log_path = 'SSL_GAN_log_' + task_name + '.csv' # Don't worry about log_path, it is named after the task
    log_path_baseline = 'baseline_log.csv'
    model_path =('savedModels/GAN')
    baseline_path = 'savedModels/baseline'
    
    tf.disable_eager_execution()
   # tf.debugging.set_log_device_placement(True)
    # train Semi-Supervised Learning GAN
    train_D_losses, train_G_losses, train_Accs = [], [], []
    val_D_losses, val_G_losses, val_Accs, TPRs = [], [], [], []
    
    cv_size = batch_size
    num_train_exs = numTrain
    num_val_exs = numTest
    print(batch_size)
    print("num_train_exs: ", num_train_exs)
    print("num_val_exs: ", num_val_exs)

    tf.reset_default_graph()

    x = tf.placeholder(tf.int32, name = 'x', shape = [None, sequence_length])
    label = tf.placeholder(tf.float32, name = 'label', shape = [None, num_classes]) # one hot label--BZ, August, 2019
    labeled_mask = tf.placeholder(tf.float32, name = 'labeled_mask', shape = [None])
    z = tf.placeholder(tf.float32, name = 'z', shape = [None, latent_size])#one 1-d noise vector per training example
    dropout_rate = tf.placeholder(tf.float32, name = 'dropout_rate')
    is_training = tf.placeholder(tf.bool, name = 'is_training')
    G_learning_rate = tf.placeholder(tf.float32, name = 'G_learning_rate')
    D_learning_rate = tf.placeholder(tf.float32, name = 'D_learning_rate')

    model = build_model(x, z, label, dropout_rate, is_training, print_summary = False)
    #    return D_real_features, D_real_logit, D_real_prob, D_fake_features, D_fake_logit, D_fake_prob
    D_real_features, D_real_logit, D_real_prob, D_fake_features, D_fake_logit, D_fake_prob, fake_data,\
    rf_real, rf_fake = model
    extended_label = prepare_labels(label) #is only for real image data
    loss_acc  = loss_accuracy(D_real_features, D_real_logit, D_real_prob,
                              D_fake_features, D_fake_logit, D_fake_prob,
                              extended_label, labeled_mask)
    _, _, _, D_L, G_L, accuracy, correct_prediction = loss_acc
    D_optimizer, G_optimizer = optimizer(D_L, G_L, G_learning_rate, D_learning_rate)

    
#     validation_generator = get_batch(data_path, label_path, num_val_exs, num_train_exs, True)

    print('training....')

    with tf.Session() as sess:       
        sess.run(tf.global_variables_initializer())
        #mnist_set = get_data()

        t_total = 0
        #changed to iterating on number of epochs!
        iter_count = 0
        iter_since_last_val = 0
        for epoch in range(num_epochs):
            for iteration in range(int(numTrain/(batch_size*(num_classes+c_ul)/(num_classes+1+c_ul)))):
            #batch_num = 0 #added--BZ Nov 2, 2019
            #training_generator = get_training_batch_and_labeled_mask(X_train, y_train, LABELED_MASK, batch_size);
                train_batch_x, train_batch_y, train_batch_mask = get_balance_train_batch_3(train_data_by_class, train_label_by_class, batch_size, c_ul)            #for train_batch_x, train_batch_y, train_batch_mask in training_generator:
                t_start = time.time()
                #batch_z = np.random.uniform(-1.0, 1.0, size = (batch_size, latent_size)) #
                #batch_z = np.random.normal(0, 1, size = (int(batch_size/num_classes), latent_size))
                batch_z = np.random.normal(0, 1, size = (int(batch_size/(num_classes+1+c_ul)), latent_size))
                #function is to be modified as labeled mask should stay with "labeled" images which are shuffled BZ--August, 2019
                #mask = get_training_batch_and_labeled_mask(XTrain, yTrain, labeledMask, batchSize);--marked as solved BZ, Nov, 2019
                train_feed_dictionary = {x: train_batch_x,
                                         z: batch_z,
                                         label: train_batch_y,
                                         labeled_mask: train_batch_mask,
                                         dropout_rate: 0.25,
                                         G_learning_rate: 5e-4,
                                         D_learning_rate: 5e-4,
                                         is_training: True}

                D_optimizer.run(feed_dict = train_feed_dictionary)
                G_optimizer.run(feed_dict = train_feed_dictionary)

                train_D_loss = D_L.eval(feed_dict = train_feed_dictionary)
                train_G_loss = G_L.eval(feed_dict = train_feed_dictionary)
                train_accuracy = accuracy.eval(feed_dict = train_feed_dictionary)
                t_total += (time.time() - t_start)

                # Debug:
                print('train_D_loss:')
                print(train_D_loss)
                print('train_G_loss:')
                print(train_G_loss)
                print('===============')
        
                train_D_losses.append(train_D_loss)
                train_G_losses.append(train_G_loss)
                train_Accs.append(train_accuracy)
                #batch_num = batch_num + 1;
            # Validation at the end of each epoch--BZ, Nov, 2019

            
            test_generator= get_test_batch(test_data_by_class, test_label_by_class, batch_size)
            val_correct_preds = []
            predictions = []
            labels = np.zeros((0, num_classes))
            for test_batch_x, test_batch_y in test_generator:
            #test_batch_generator = get_test_batch(X_test, y_test, batch_size);#added--BZ, Nov 2, 2019
            #for test_batch_x, test_batch_y in test_batch_generator:
                val_batch_z = np.random.normal(0, 1, size = (len(test_batch_y), latent_size))
                mask = np.ones(len(test_batch_y));#all test data is labeled   added--BZ, Nov 2, 2019
                val_feed_dictionary = {x: test_batch_x,
                                       z: val_batch_z,
                                       label: test_batch_y,
                                       labeled_mask: mask,
                                       dropout_rate: 0.0,
                                       is_training: False}


                val_D_loss = D_L.eval(feed_dict = val_feed_dictionary)
                val_G_loss = G_L.eval(feed_dict = val_feed_dictionary)

                val_correct_pred = correct_prediction.eval(feed_dict = val_feed_dictionary)
                val_correct_preds = np.concatenate((val_correct_preds, val_correct_pred))
                    
                val_D_real_prob = D_real_prob.eval(feed_dict = val_feed_dictionary)
                predictions = np.concatenate((predictions, np.argmax(val_D_real_prob[:, :-1], axis = 1)))
                labels = np.concatenate((labels, test_batch_y))
            
            val_accuracy = compute_val_accuracy(val_correct_preds)
            val_Accs.append(val_accuracy)
    
            CM = confusion_matrix(np.argmax(labels, axis = 1), predictions)
            sum0 = np.sum(CM[0,:]) # Row sum
            sum1 = np.sum(CM[1,:])
            sum2 = np.sum(CM[2,:])
            #TPR = CM[1][1]/(CM[1][0]+CM[1][1])
            #TNR = CM[0][0]/(CM[0][0]+CM[0][1])
            #TPRs.append(TPR)
            #print(val_correct_preds);
            #print('validation_acc: %f' %(val_accuracy))
            log_loss_acc_diagonals(log_path, epoch, train_D_loss, train_G_loss, train_accuracy,
                 val_accuracy, CM[0][0], CM[1][1], CM[2][2], CM[3][3], 'w')
    
            save_model_recalls(model_path, sess, epoch, val_accuracy, val_Accs, CM[0][0]/sum0, CM[1][1]/sum1, CM[2][2]/sum2)
            
            fakes = fake_data.eval(feed_dict = val_feed_dictionary)
            print('epoch done')
            #print(fakes)
            #save_fake_image(fakes, epoch)
            #confusion matrix
            #print('epoch'+str(epoch)+' CM:')
            #print(confusion_matrix(np.argmax(labels, axis = 1), predictions, normalize = 'true'))
                
        
    return train_D_losses, train_G_losses, train_Accs, val_Accs

def train_gan():
    global numTrain
    global numTest 
    loadClasses()
            # Select data as unlabeled:
    for i in range(num_classes):
        print(train_label_by_class)
        numInClass = len(train_label_by_class[i])
        #mask =  np.concatenate((np.ones(int(numInClass * labeled_rate), dtype='int'), np.zeros(numInClass - int(numInClass * labeled_rate), dtype='int')), axis=0)
        num_unlabeled = int(numInClass-numInClass*labeled_rate)
        train_data_class = train_data_by_class[i]
        train_label_class = train_data_by_class[i]
        indices = [i for i in range(numInClass)]
        np.random.shuffle(indices)
        unlabeled_is, labeled_is = np.split(indices,[num_unlabeled]) # Unlabeled and labeled indices

        # Concatenate image data array
        #unlabeled_data = np.concatenate((unlabeled_data, train_data_by_class[i][unlabeled_is]))
        #labeled_data_baseline = np.concatenate((labeled_data_baseline, train_data_by_class[i][labeled_is]))

        #unlabeled_indices_by_class.append(unlabeled_is)
        labeled_indices_by_class.append(labeled_is)

    #np.save(unlabeled_data, 'numpyData/0.25_labeled_data')

    numTrain = sum([len(c) for c in train_label_by_class])
    numTest = sum([len(c) for c in test_label_by_class])
    log_path = 'SSL_GAN_log_' + task_name + '.csv' # Don't worry about log_path, it is named after the task
    log_path_baseline = 'baseline_log.csv'
    model_path ='savedModels/GAN'
    baseline_path = 'savedModels/baseline'

    train_SSL_GAN(16, 10, train_data_by_class, train_label_by_class, test_data_by_class, test_label_by_class)