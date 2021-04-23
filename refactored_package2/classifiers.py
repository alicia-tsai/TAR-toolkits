import numpy as np 
from classify_linear import *
from nlp_preprocess import * 
from cnn_model import * 
from gan_model import * 
from run_models import * 

def run_classifiers(train, test, output, trainYes = False): 
    """
    README: This function runs the Logistic Regression, SVM, GAN and CNN models on the given training and test data, runs a majority vote on the results, and saves the output to a csv entitled "output.csv"
    
    train: the filename of the csv file with training data 
    

    test: the filename of the csv file with test data
    """
    test_frame = pd.read_csv(test) 
    test_frame = test_frame.dropna() 
    test_frame['cleaned'] = clean(test_frame['sentence'])
  #  print(test_frame['cleaned']) 
   # print(test_frame) 
    test_frame['lemmatized'] = lemmatize(test_frame['cleaned'])  
    
    print(test_frame) 
    regression_return = run_classifier(train, "lr")
    svm_return = run_classifier(train, "svm")
    
    regression_model = regression_return[1] 
    svm_model = svm_return[1]
    
    bow_converter = regression_return[0]
    
    x_test = bow_converter.transform(test_frame['lemmatized'])
    
    
    test_frame["svm"] = svm_model.predict(x_test) 
    test_frame["lr"] = regression_model.predict(x_test) 
    
    preprocess_data(train, test) 
	
    print("preprocessed") 
    
    if(trainYes): 
        train_gan()
        train_cnn()


    gan_predictions = (run_GAN())
   # print("Rangan")
    cnn_predictions = (run_CNN())

    gan_label = [] 
    cnn_label = [] 
    
    for x in range(0, len(gan_predictions)): 
        if gan_predictions[x] == 0: 
            gan_label.append("Building") 
        if gan_predictions[x] == 1: 
            gan_label.append("Infrastructure") 
        if gan_predictions[x] == 2: 
            gan_label.append("Resilience") 
        if gan_predictions[x] == 3: 
            gan_label.append("Other") 
            
        if cnn_predictions[x] == 0: 
            cnn_label.append("Building") 
        if cnn_predictions[x] == 1: 
            cnn_label.append("Infrastructure") 
        if cnn_predictions[x] == 2: 
            cnn_label.append("Resilience") 
        if cnn_predictions[x] == 3: 
            cnn_label.append("Other") 
            
    test_frame["gan"] = gan_label 
    test_frame["cnn"] = cnn_label 
    
    test_frame = test_frame.drop("cleaned", axis = 1) 
    test_frame = test_frame.drop("lemmatized", axis = 1) 
    
    regression = get_column_vector(regression_model.predict(x_test)) 
    svm = get_column_vector(svm_model.predict(x_test)) 
    gan_label = get_column_vector(gan_label) 
    cnn_label = get_column_vector(cnn_label) 
    
    majority_label = run_majority_vote([regression, svm, gan_label, cnn_label])
    
    test_frame["majority"] = majority_label 
    
    test_frame.to_csv(output) 
    
	
def get_column_vector(inputList): 
    return np.reshape(inputList, (len(inputList), 1)) 

def lower(inputString): 
    return inputString.lower() 

def run_majority_vote(predictions): 
    retLabels = [] 
    
    combined = np.hstack([predictions[x] for x in range(0, len(predictions))]) 
    
    for x in range(0, len(combined)): 
        currentRow = combined[x] 
        
        frequencyTrack = {"building": 0, "resilience": 0, "infrastructure": 0, "other": 0}
        
        for j in range(0, len(currentRow)): 
            frequencyTrack[lower(currentRow[j])] += 1
            
        retLabels.append(max(frequencyTrack, key = frequencyTrack.get))
    
    return retLabels
            

if __name__ == '__main__': 
    run_classifiers(sys.argv[1], sys.argv[2], "output.csv", False) 
    #run_classifiers("dataset.csv", "albania.csv") 