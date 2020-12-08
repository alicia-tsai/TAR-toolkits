# Import and run a trained model 
import numpy as np
from os import listdir
import tensorflow.compat.v1 as tf
from sklearn.metrics import confusion_matrix
import pandas as pd

def get_test_batch(XTest, yTest, batchSize):
    #a generator that slices and returns batchSize of XTrain and yTrain instances from top down
    counter = 0;
    while True:
        if counter >= len(yTest):
            break;
        returnXTest = XTest[counter:counter + batchSize];
        returnYTest = yTest[counter:counter + batchSize];
        counter = counter + batchSize;
		
		
        yield returnXTest, returnYTest
		

  
#y_test_dummy = np.zeros((260, 3))
y_test_dummy = np.zeros((260, 3)) 

batch_size = 260
latent_size = 100
sequence_length = 64
num_classes = 3

def run_GAN(): 
   # tf.contrib.rnn
    raw_x_test = np.load('numpyData/raw_x_test.npy')
    raw_y_test = np.load('numpyData/raw_y_test.npy')
       
    directoryList = listdir("savedModels/GAN/NLP_embedding_80") 
    
    metaFiles = filter(lambda x: x[-10:] == ".ckpt.meta", directoryList)
    metaCopy = list(metaFiles) 
    metaFilesEpoch = map(lambda x: int(x[x.index("epoch") + 6: -18]), metaCopy) 
    
    metaFilesEpoch = (list(metaFilesEpoch)) 
    maxIndex = (np.argmax(list(metaFilesEpoch)))

    chosenFile = "savedModels/GAN/NLP_embedding_80/" + list(metaCopy)[maxIndex]
   
    
    tf.reset_default_graph()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(chosenFile)
        saver.restore(sess, chosenFile[:-5])

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('x:0')
        z = graph.get_tensor_by_name('z:0')
        label = graph.get_tensor_by_name('label:0')
        labeled_mask = graph.get_tensor_by_name('labeled_mask:0')
        dropout_rate = graph.get_tensor_by_name('dropout_rate:0')
        is_training = graph.get_tensor_by_name('is_training:0')

        logits = graph.get_tensor_by_name('Discriminator/scores:0')
        conv = graph.get_tensor_by_name('Discriminator/conv:0')
        pool = graph.get_tensor_by_name('Discriminator/pool:0') 

        test_logits_all = []
        test_label_all = []
		
        for batch_x, batch_y in get_test_batch(raw_x_test, y_test_dummy, batch_size):
            mask = np.ones(len(batch_y)) # Mark all as labeled
            batch_z = np.random.normal(0, 1, size = (len(batch_y), latent_size))

            test_feed_dictionary = {x: batch_x,
									z: batch_z,
									label: batch_y,
									labeled_mask: mask,
									dropout_rate: 0.0,
									is_training: False}
            test_logits = logits.eval(feed_dict=test_feed_dictionary)
            print('conv shape')
            print(conv.shape)
            print('pool shape')
            print(pool.shape)

            test_logits_all.append(test_logits)
            test_label_all.append(batch_y)
        
        test_logits_all = np.concatenate(test_logits_all)
        test_label_all = np.concatenate(test_label_all)
        preds = np.argmax(test_logits_all, axis=1)

    return preds 

def run_CNN(): 
    raw_x_test = np.load('numpyData/raw_x_test.npy')
    raw_y_test = np.load('numpyData/raw_y_test.npy')
    
    tf.reset_default_graph()
        
    directoryList = listdir("savedModels/CNN/CNN_80/") 
    
    metaFiles = filter(lambda x: x[-10:] == ".ckpt.meta", directoryList)
    metaCopy = list(metaFiles) 
    metaFilesEpoch = map(lambda x: int(x[x.index("epoch") + 6: -18]), metaCopy) 
    
    metaFilesEpoch = (list(metaFilesEpoch)) 
    maxIndex = (np.argmax(list(metaFilesEpoch)))

    chosenFile = "savedModels/CNN/CNN_80/" + list(metaCopy)[maxIndex]
  
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(chosenFile)
        saver.restore(sess, chosenFile[: -5]) 
            
        graph = tf.get_default_graph()
        # Inputs
        input_x = graph.get_tensor_by_name('input_x:0')
        input_y = graph.get_tensor_by_name('input_y:0')
        dropout_keep_prob = graph.get_tensor_by_name('dropout_keep_prob:0')
        # Output of interest
        predictions = graph.get_tensor_by_name('output/predictions:0')

        test_predictions_all = []

        for batch_x, batch_y in get_test_batch(raw_x_test, y_test_dummy, batch_size):
            test_feed_dictionary = val_feed_dict = {
                        input_x: batch_x,
                        input_y: batch_y,
                        dropout_keep_prob: 1.0 # No dropout
                    }
            test_predictions = predictions.eval(feed_dict=test_feed_dictionary)
            test_predictions_all.append(test_predictions)

        test_predictions_all = np.concatenate(test_predictions_all)

    return test_predictions_all

		
