import librosa
import soundfile
import os, glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#Load libraries needed for classification 
from sklearn.pipeline import Pipeline
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

observed_emotions=['calm', 'happy', 'fearful', 'disgust']

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result



def load_data(test_size=0.2):
    x,y=[],[]
 #  counter=0
    for file in glob.glob("D:\\Study\\Diploma_graduation_project\\Speech_Project\\speech-emotion-recognition-ravdess-data\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
 #      print(file_name)
 #      counter=counter+1
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
 #      print(feature)
        x.append(feature)
        y.append(emotion)
 #  print(counter)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

x_train,x_test,y_train,y_test=load_data(test_size=0.25)
##X_TRAIN=pd.DataFrame(x_train)
#X_TRAIN=[str (item) for item in X_TRAIN]
#X_TRAIN= [item for item in X_TRAIN if not isinstance(item, int)]
##Y_TRAIN=pd.DataFrame(y_train)
#Y_TRAIN=[str (item) for item in Y_TRAIN]
#Y_TRAIN= [item for item in Y_TRAIN if not isinstance(item, int)]
#print(X_TRAIN)
#Get the shape of the training and testing datasets
#print((x_train.shape[0], x_test.shape[0])) 

model = [
            "CountVectorizer + Naïve Bayes Multinomial", 
            "TFIDFVectorizer + Naïve Bayes Multinomial", 
            "CountVectorizer with uni-grams and bi-grams + Naïve Bayes Multinomial", 
            "CountVectorizer + Logistic Regression", 
            "TFIDFVectorizer + Logistic Regression", 
            "CountVectorizer with uni-grams and bi-grams + Logistic Regression",
            "TFIDFVectorizer + SVM (Linear Kernel)",
            "CountVectorizer + SVM (Linear Kernel)",
            "CountVectorizer with uni-grams and bi-grams + SVM (Linear Kernel)",
            "TFIDFVectorizer + SVM (RBF)",
            "CountVectorizer + SVM (RBF)",
            "CountVectorizer with uni-grams and bi-grams + SVM (RBF)",
            "CountVectorizer + Random Forest",
            "MLPClassifier"
        ]
#intialize the output matrix
#result = pd.DataFrame(columns=['Accuracy', 'FScore'])
kfold = KFold(n_splits=10, shuffle=True, random_state=1234)
for i in model:
    if i == 'CountVectorizer + Naïve Bayes Multinomial':
        #CountVectorizer + Naïve Bayes Multinomial pipeline
        pipeline = Pipeline([
        #('CountVectprizer', CountVectorizer()),
        ('naive_bayes_Multinomial', naive_bayes.GaussianNB())
        ])
    elif i == 'TFIDFVectorizer + Naïve Bayes Multinomial':
        #TFIDFVectorizer + Naïve Bayes Multinomial pipeline
        pipeline = Pipeline([
        #('TFIDFVectprizer', TfidfVectorizer()),
        ('naive_bayes_Multinomial', naive_bayes.GaussianNB())
        ])
    elif i == 'CountVectorizer with uni-grams and bi-grams + Naïve Bayes Multinomial':
        #CountVectorizer with uni-grams and bi-grams + Naïve Bayes Multinomial pipeline
        pipeline = Pipeline([
        #('CountVectorizer', CountVectorizer(ngram_range=(1,2))),
        ('naive_bayes_Multinomial', naive_bayes.GaussianNB())
        ])
    elif i == 'CountVectorizer + Logistic Regression':
        #CountVectorizer + Logistic Regression pipeline
        pipeline = Pipeline([
        #('CountVectorizer', CountVectorizer()),
        ('LogisticRegression', LogisticRegression())
        ])
    elif i == 'TFIDFVectorizer + Logistic Regression':
        #TFIDFVectorizer + Logistic Regression pipeline
        pipeline = Pipeline([
        #('TFIDFVectorizer', TfidfVectorizer()),
        ('LogisticRegression', LogisticRegression())
        ])
    elif i == 'CountVectorizer with uni-grams and bi-grams + Logistic Regression':
        #CountVectorizer with uni-grams and bi-grams + Logistic Regression pipeline
        pipeline = Pipeline([
        #('CountVectorizer', CountVectorizer(ngram_range=(1,2))),
        ('LogisticRegression', LogisticRegression())
        ])
    elif i == 'CountVectorizer + SVM (Linear Kernel)':
        #CountVectorizer + SVM (Linear Kernel) pipeline
        pipeline = Pipeline([
        #('CountVectorizer', CountVectorizer()),
        ('SVM_linear_kernel', svm.LinearSVC())
        ])
    elif i == 'TFIDFVectorizer + SVM (Linear Kernel)':
        #TFIDFVectorizer + SVM (Linear Kernel) pipeline
        pipeline = Pipeline([
        #('TFIDFVectorizer', TfidfVectorizer()),
        ('SVM_linear_kernel', svm.LinearSVC())
        ])
    elif i == 'CountVectorizer with uni-grams and bi-grams + SVM (Linear Kernel)':
        #CountVectorizer with uni-grams and bi-grams + SVM (Linear Kernel) pipeline
        pipeline = Pipeline([
        #('CountVectorizer', CountVectorizer(ngram_range=(1,2))),
        ('SVM_linear_kernel', svm.LinearSVC())
        ])
    #elif i == 'CountVectorizer + SVM (RBF)':
    #    #CountVectorizer + SVM (RBF) pipeline
    #    pipeline = Pipeline([
    #    #('CountVectorizer', CountVectorizer()),
    #    ('SVM_RBF', svm.SVR(kernel='rbf'))
    #    ])
    #elif i == 'TFIDFVectorizer + SVM (RBF)':
    #    #TFIDFVectorizer + SVM (RBF) pipeline
    #    pipeline = Pipeline([
    #    #('TFIDFVectorizer', TfidfVectorizer()),
    #    ('SVM_RBF', svm.SVR(kernel='rbf'))
    #    ])
    #elif i == 'CountVectorizer with uni-grams and bi-grams + SVM (RBF)':
    #    #CountVectorizer with uni-grams and bi-grams + SVM (RBF) pipeline
    #    pipeline = Pipeline([
    #    #('CountVectorizer', CountVectorizer(ngram_range=(1,2))),
    #    ('SVM_RBF', svm.SVR(kernel='rbf'))
    #    ])
    elif i == 'CountVectorizer + Random Forest':
        #CountVectorizer + Random Forest pipeline
        pipeline = Pipeline([
         #('CountVectorizer', CountVectorizer()),
         ('RandomForest', RandomForestClassifier())
         ])
    elif i == 'MLPClassifier':
        #MLPClassifier
        pipeline = Pipeline([
        ('MLPClassifier', MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500))
        ])    
    else:
        print("ERROR: Model Not Supported!")
        continue
           
    accuracy = 0
    
    #for fold, (train_index, val_index) in enumerate(kfold.split(X_TRAIN, y_train)):
     #   train_x, train_y = X_TRAIN.iloc[train_index], Y_TRAIN.iloc[train_index]
     #   val_x, val_y = X_TRAIN.iloc[val_index], Y_TRAIN.iloc[val_index]
        
        #Model fit & Prediction
     #   pipeline.fit(train_x, train_y)
     #   predictions = pipeline.predict(val_x)
        
        #Calculate the Accuracy
     #   accuracy += accuracy_score(val_y, predictions.round())

    #accuracy /= kfold.get_n_splits()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    accuracy=accuracy_score(y_true=y_test, y_pred=predictions)
    print(i + ":")
    print("Accuracy = {}".format(accuracy.round(2)))
    


      