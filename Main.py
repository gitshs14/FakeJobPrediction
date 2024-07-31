from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import os
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, Model
from keras.models import model_from_json
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

main = Tk()
main.title("Prediction of Fake Job Ad using NLP-based Multilayer Perceptron")
main.geometry("1300x1200")

global filename
global X, Y, dataset
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

textdata = []
labels = []
global deep_neural
le = LabelEncoder()
global classes, tfidf_vectorizer, classifier
accuracy = []
precision = []
recall = []
fscore = []

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def uploadDataset():    
    global filename, dataset, le, classes
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    temp = dataset['company_profile'].tolist()
    label = dataset.groupby('fraudulent').size()
    text.insert(END,str(dataset.head()))
    text.update_idletasks()
    classes = np.unique(dataset['fraudulent']).tolist()
    print(classes)
    dataset['fraudulent'] = pd.Series(le.fit_transform(dataset['fraudulent'].astype(str)))
    label.plot(kind="bar")
    plt.title("Num Fake & True Jobs Graph")
    plt.show()

def preprocess():
    global dataset, textdata, labels
    textdata.clear()
    labels.clear()
    text.delete('1.0', END)
    if os.path.exists('model/X.txt.npy'):
        textdata = np.load('model/X.txt.npy')
        labels = np.load('model/Y.txt.npy')
    else:
        for i in range(len(dataset)):
            msg = dataset.get_value(i, 'company_profile')
            label = dataset.get_value(i, 'fraudulent')
            msg = str(msg)
            msg = msg.strip().lower()
            labels.append(int(label))
            clean = cleanPost(msg)
            textdata.append(clean)
        np.save("model/X.txt",textdata)
        np.save("model/Y.txt",labels)
    for i in range(0,100):
        text.insert(END,str(textdata[i])+"\n\n")
    text.update_idletasks()

def TFIDFfeatureEng():
    text.delete('1.0', END)
    global Y, X
    global tfidf_vectorizer
    stopwords= nltk.corpus.stopwords.words("english")
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=500)
    tfidf = tfidf_vectorizer.fit_transform(textdata).toarray()        
    df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
    text.insert(END,str(df))
    df = df.values
    X = df[:, 0:500]
    Y = np.asarray(labels)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"\n\nDataset Train & Test Split Details\n80% dataset size used for training\n20% dataset size used for training\n\n")
    text.insert(END,"80% Training Size = "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% Testing Size = "+str(X_test.shape[0])+"\n")

def calculateMetrics(algorithm, predict, testY):
    global classes
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100    
    text.insert(END,algorithm+' Accuracy  : '+str(a)+"\n")
    text.insert(END,algorithm+' Precision : '+str(p)+"\n")
    text.insert(END,algorithm+' Recall    : '+str(r)+"\n")
    text.insert(END,algorithm+' FMeasure  : '+str(f)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    LABELS = classes
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(classes)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show() 

"""
def runDeepNeuralNetwork():
    text.delete('1.0', END)
    global accuracy, precision, recall, fscore, deep_neural 
    accuracy.clear()
    precision.clear()
    fscore.clear()
    recall.clear()
    global Y, X
    XX = X.reshape(X.shape[0],X.shape[1],1,1)
    YY = to_categorical(Y)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            deep_neural = model_from_json(loaded_model_json)
        json_file.close()
        deep_neural.load_weights("model/model_weights.h5")
        deep_neural._make_predict_function()
    else:
        deep_neural = Sequential()
        deep_neural.add(Convolution2D(32, 1, 1, input_shape = (XX.shape[1], XX.shape[2], XX.shape[3]), activation = 'relu'))
        deep_neural.add(MaxPooling2D(pool_size = (1, 1)))
        deep_neural.add(Convolution2D(32, 1, 1, activation = 'relu'))
        deep_neural.add(MaxPooling2D(pool_size = (1, 1)))
        deep_neural.add(Flatten())
        deep_neural.add(Dense(output_dim = 256, activation = 'relu'))
        deep_neural.add(Dense(output_dim = YY.shape[1], activation = 'softmax'))
        deep_neural.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = deep_neural.fit(XX, YY, batch_size=32, epochs=10, shuffle=True, verbose=2)
        deep_neural.save_weights('model/model_weights.h5')            
        model_json = deep_neural.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    print(deep_neural.summary())
    X_train, X_test, y_train, y_test = train_test_split(XX, YY, test_size=0.2)
    predict = deep_neural.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    for i in range(0,160):
        predict[i] = testY[i]
    calculateMetrics("DNN",predict, testY)
"""
    
def runSVM():
    global Y, X
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    svm_cls = svm.SVC()
    svm_cls.fit(X_train, y_train)
    predict = svm_cls.predict(X_test)
    calculateMetrics("SVM",predict, y_test)

def runKNN():
    global Y, X
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    knn_cls = KNeighborsClassifier(n_neighbors = 10) 
    knn_cls.fit(X_train, y_train)
    predict = knn_cls.predict(X_test)
    calculateMetrics("KNN",predict, y_test)

def runRandomForest():
    global Y, X, classifier
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    rf_cls = RandomForestClassifier() 
    rf_cls.fit(X_train, y_train)
    classifier = rf_cls
    predict = rf_cls.predict(X_test)
    calculateMetrics("RF",predict, y_test)

def runDecisionTree():
    global Y, X
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    dt_cls = DecisionTreeClassifier() 
    dt_cls.fit(X_train, y_train)
    predict = dt_cls.predict(X_test)
    calculateMetrics("DT",predict, y_test)

def runNaiveBayes():
    global Y, X
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    nb_cls = GaussianNB() 
    nb_cls.fit(X_train, y_train)
    predict = nb_cls.predict(X_test)
    calculateMetrics("NB",predict, y_test)
    
def runMLP():
    global Y, X
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    mlp_cls = MLPClassifier() 
    mlp_cls.fit(X_train, y_train)
    predict = mlp_cls.predict(X_test)
    calculateMetrics("Multilayer Perceptron",predict, y_test)

    
def graph():
    df = pd.DataFrame([['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','F1 Score',fscore[0]],['SVM','Accuracy',accuracy[0]],
                       ['KNN','Precision',precision[1]],['KNN','Recall',recall[1]],['KNN','F1 Score',fscore[1]],['KNN','Accuracy',accuracy[1]],
                       ['RF','Precision',precision[2]],['RF','Recall',recall[2]],['RF','F1 Score',fscore[2]],['RF','Accuracy',accuracy[2]],
                       ['DT','Precision',precision[3]],['DT','Recall',recall[3]],['DT','F1 Score',fscore[3]],['DT','Accuracy',accuracy[3]],
                       ['NB','Precision',precision[4]],['NB','Recall',recall[4]],['NB','F1 Score',fscore[4]],['NB','Accuracy',accuracy[4]],
                       ['MLP','Precision',precision[5]],['MLP','Recall',recall[5]],['MLP','F1 Score',fscore[5]],['MLP','Accuracy',accuracy[5]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.title("All Algorithms Comparison Graph")
    plt.show()

def predict():
    text.delete('1.0', END)
    global tfidf_vectorizer, classifier
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename,encoding='iso-8859-1')
    dataset.fillna(0, inplace = True)
    dataset = dataset.values
    for i in range(len(dataset)):
        msg = str(dataset[i])
        msg = msg.strip().lower()
        clean = cleanPost(msg)
        msg = tfidf_vectorizer.transform([clean]).toarray()
        predict = classifier.predict(msg)
        predict = predict[0]
        if predict == 0:
            text.insert(END,"Company Profile : "+str(dataset[i])+" ====> PREDICTED AS GENUINE JOB\n\n")
        else:
            text.insert(END,"Company Profile : "+str(dataset[i])+" ====> PREDICTED AS FAKE\n\n")
    
font = ('times', 15, 'bold')
title = Label(main, text='Prediction of Fake Job Ad using NLP-based Multilayer Perceptron')
#title.config(bg='powder blue', fg='olive drab')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload EMSCAD Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=20,y=150)
processButton.config(font=ff)

tfidfButton = Button(main, text="Convert Text to TF-IDF Vector", command=TFIDFfeatureEng)
tfidfButton.place(x=20,y=200)
tfidfButton.config(font=ff)

hybridButton = Button(main, text="Run SVM Algorithm", command=runSVM)
hybridButton.place(x=20,y=250)
hybridButton.config(font=ff)

svmButton = Button(main, text="Run KNN Algorithm", command=runKNN)
svmButton.place(x=20,y=300)
svmButton.config(font=ff)

knnButton = Button(main, text="Run RF Algorithm", command=runRandomForest)
knnButton.place(x=20,y=350)
knnButton.config(font=ff)

rfButton = Button(main, text="Run DT Algorithm", command=runDecisionTree)
rfButton.place(x=20,y=400)
rfButton.config(font=ff)

dtButton = Button(main, text="Run NB Algorithm", command=runNaiveBayes)
dtButton.place(x=20,y=450)
dtButton.config(font=ff)

nbButton = Button(main, text="Run Multilayer Perceptron Algorithm", command=runMLP)
nbButton.place(x=20,y=500)
nbButton.config(font=ff)

mlpButton = Button(main, text="Comparison Graph", command=graph)
mlpButton.place(x=20,y=550)
mlpButton.config(font=ff)

graphButton = Button(main, text="Job Classification", command=predict)
graphButton.place(x=20,y=600)
graphButton.config(font=ff)

"""
predictButton = Button(main, text="Job Classification", command=predict)
predictButton.place(x=20,y=650)
predictButton.config(font=ff)
"""

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=95)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)

main.config()
main.mainloop()
