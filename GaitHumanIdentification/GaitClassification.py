from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter

main = tkinter.Tk()
main.title("Human Identification From Freestyle Walks Using Posture-Based Gait Feature")
main.geometry("1200x1200")

global filename
proto_File = "model/pose_deploy_linevec.prototxt"
weights_File = "model/pose_iter_440000.caffemodel"
n_Points = 18
POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
in_Width = 368
in_Height = 368
threshold = 0.1
global net
POSE_NAMES = ["Head", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
              "RAnkle", "LHip", "LKnee", "LAnkle", "Chest", "Background"]
global dataset, X, Y, X_train, X_test, y_train, y_test, accuracy, precision, recall, fscore, scaler
global labels, et_cls

def uploadDataset():
    global filename, dataset, labels
    global net
    filename = filedialog.askopenfilename(initialdir="Dataset")
    net = cv2.dnn.readNetFromCaffe(proto_File, weights_File)
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    dataset = dataset.values
    text.insert(END,str(dataset))
    text.update_idletasks()
    labels = ['Person 1', 'Person 2', 'Person 3', 'Person 4']
    names, count = np.unique(Y, return_counts = True)
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (4, 3)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Dataset Class Label Graph")
    plt.ylabel("Count")
    plt.show()

def preprocessDataset():
    global dataset, X, Y, scaler
    text.delete('1.0', END)
    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)#shuffling dataset
    X = X[indices]
    Y = Y[indices]
    scaler = MinMaxScaler((0, 1))
    X = scaler.fit_transform(X)
    text.insert(END,"Dataset Shuffling & Normalization Complated\n\n")
    text.insert(END,"Normalized Dataset Values = "+str(X))

def trainTestSplit():
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train & Test Split Details\n\n")
    text.insert(END,"Total records found in Dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"80% records used to train algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% records used to test algorithms : "+str(X_test.shape[0])+"\n")

#function to calculate various metrics such as accuracy, precision etc
def calculateMetrics(algorithm, predict, testY):
    global labels
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100     
    text.insert(END,algorithm+' Accuracy  : '+str(a)+"\n")
    text.insert(END,algorithm+' Precision   : '+str(p)+"\n")
    text.insert(END,algorithm+' Recall      : '+str(r)+"\n")
    text.insert(END,algorithm+' FMeasure    : '+str(f)+"\n\n")    
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(4, 3)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    

def trainTree():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore, et_cls
    text.delete('1.0', END)
    accuracy = []
    precision = []
    recall = []
    fscore = []
    #training Extra Tree Classifier on training data
    et_cls = DecisionTreeClassifier()
    et_cls.fit(X_train, y_train)
    #call this function to predict on test data
    predict = et_cls.predict(X_test)
    #call this function to calculate accuracy and other metrics
    calculateMetrics("Extra Tree", predict, y_test)

def trainKNN():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    #call this function to predict on test data
    predict = knn.predict(X_test)
    #call this function to calculate accuracy and other metrics
    calculateMetrics("KNN", predict, y_test)

def trainMLP():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    mlp = MLPClassifier(max_iter=800)
    mlp.fit(X_train, y_train)
    #call this function to predict on test data
    predict = mlp.predict(X_test)
    #call this function to calculate accuracy and other metrics
    calculateMetrics("MLP", predict, y_test)

def graph():
    global accuracy, precision, recall, fscore
    df = pd.DataFrame([['Extra Tree','Accuracy',accuracy[0]],['Extra Tree','Precision',precision[0]],['Extra Tree','Recall',recall[0]],['Extra Tree','FSCORE',fscore[0]],
                       ['KNN','Accuracy',accuracy[1]],['KNN','Precision',precision[1]],['KNN','Recall',recall[1]],['KNN','FSCORE',fscore[1]],
                       ['MLP','Accuracy',accuracy[2]],['MLP','Precision',precision[2]],['MLP','Recall',recall[2]],['MLP','FSCORE',fscore[2]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', figsize=(8, 3))
    plt.title("All Algorithms Performance Graph")
    plt.show()

def recognizeHuman():
    text.delete('1.0', END)
    global et_cls, scaler, labels
    filename = filedialog.askopenfilename(initialdir="Videos")
    cap = cv2.VideoCapture(filename)
    while True:
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame, (500, 500))
            frameCopy = np.copy(frame)
            frame_Width = frame.shape[1]
            frame_Height = frame.shape[0]
            img = np.zeros((frame_Height,frame_Width,3), dtype=np.uint8)
            inp_Blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (in_Width, in_Height), (0, 0, 0), swapRB=False, crop=False)
            net.setInput(inp_Blob)
            output = net.forward()
            H = output.shape[2]
            W = output.shape[3]
            points = []
            test_points = []
            for i in range(n_Points):
                probMap = output[0, i, :, :]
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
                x = (frame_Width * point[0]) / W
                y = (frame_Height * point[1]) / H
                if prob > threshold : 
                    cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                    points.append((int(x), int(y)))
                    temp = []
                    temp.append([x, y])
                    temp = np.asarray(temp)
                    temp = scaler.transform(temp)
                    predict = et_cls.predict(temp)
                    predict = int(predict[0])
                    print(predict)
                    test_points.append(labels[predict])
                else :
                    points.append(None)
            for pair in POSE_PAIRS:
                partA = pair[0]
                partB = pair[1]
                if points[partA] and points[partB]:
                    cv2.line(img, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
                    cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
                    cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            counter = Counter(test_points)
            print(counter)
            max_element, max_count = counter.most_common(1)[0]        
            cv2.putText(frame, "Human Recognized As "+max_element, (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
            cv2.imshow('Human Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break            
        else:
            break
    cap.release()
    cv2.destroyAllWindows()    

font = ('times', 14, 'bold')
title = Label(main, text='Human Identification From Freestyle Walks Using Posture-Based Gait Feature')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')

uploadButton = Button(main, text="Upload Kinect Gait Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=400,y=100)

preprocessButton = Button(main, text="Dataset Preprocessing", command=preprocessDataset)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1)

splitButton = Button(main, text="Dataset Train Test Split", command=trainTestSplit)
splitButton.place(x=50,y=200)
splitButton.config(font=font1)

extratreeButton = Button(main, text="Train Extra Tree Classifier", command=trainTree)
extratreeButton.place(x=50,y=250)
extratreeButton.config(font=font1)

knnButton = Button(main, text="Train KNN Classifier", command=trainKNN)
knnButton.place(x=50,y=300)
knnButton.config(font=font1)  

mlpButton = Button(main, text="Train MLP Classifier", command=trainMLP)
mlpButton.place(x=50,y=350)
mlpButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=50,y=400)
graphButton.config(font=font1)

predictButton = Button(main, text="Human Recognition Using Gait Feature", command=recognizeHuman)
predictButton.place(x=50,y=450)
predictButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=150)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()
