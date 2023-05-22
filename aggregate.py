import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def process_data(dataname):

    DATA_PATH = "dataset/"
    FEATURE = dataname

    train_neg = pd.read_csv(DATA_PATH + FEATURE + "_TR_neg_SPIDER.csv")
    train_pos = pd.read_csv(DATA_PATH + FEATURE + "_TR_pos_SPIDER.csv")
    test_neg = pd.read_csv(DATA_PATH + FEATURE + "_TS_neg_SPIDER.csv")
    test_pos = pd.read_csv(DATA_PATH + FEATURE + "_TS_pos_SPIDER.csv")

    train_frames = [train_neg, train_pos]
    test_frames = [test_neg, test_pos]

    train_df= pd.concat(train_frames)
    test_df = pd.concat(test_frames)

    columns = list(train_df.columns)
    columns.remove("druggable")
    columns.remove("seq_name")

    train_set = train_df.drop(['seq_name'] , axis=1)

    train_final = train_set.copy()
    test_final = test_df.copy()

    for column in columns:
        mean = train_set[column].mean()
        deviation = train_set[column].std()

        train_final[column] = (train_set[column] - mean) / deviation
        test_final[column] = (test_final[column] - mean) / deviation

    return([train_final,test_final])


def classify_data(dataset,classify_type):

    full_train_X = dataset.drop('druggable', axis=1)
    full_train_y = dataset['druggable']

    full_model = get_model(classify_type)

    full_model.fit(full_train_X, full_train_y)

    return(full_model)

def get_model(model_name):

    if(model_name=="Logistic Regression"):

        model = LogisticRegression(max_iter=500)
    
    elif(model_name=="Support Vector Machines"):

        model = LinearSVC(max_iter=1000)

    elif(model_name=="Decision Trees"):

        model = DecisionTreeClassifier(max_iter=500)

    elif(model_name=="Random Forest"):

        model = RandomForestClassifier(max_iter=500)
    
    elif(model_name=="Naive Bayes"):

        model = GaussianNB(max_iter=500)
    
    elif(model_name=="K-Nearest Neighbor"):

        model = KNeighborsClassifier(max_iter=500)

    return(model)

def aggregate_pipeline():
    
    dataset = process_data()
    model = classify_data(dataset[0],"Logistic Regression")

    testData = dataset[1]

    full_test_X = testData.drop(columns=['seq_name', 'druggable'], axis=1)
    full_test_y = testData['druggable']

    prediction = model.predict(full_test_X)

    sequence_data = testData['seq_name'].tolist()
    positives = ""
    negatives = ""

    accuracy = accuracy_score(full_test_y, prediction)
    sensitivity = recall_score(full_test_y, prediction)
    specificity = recall_score(full_test_y, prediction, pos_label=0)
    precision = precision_score(full_test_y, prediction)
    f1_measure = f1_score(full_test_y, prediction)

    print("Overall Accuracy :",accuracy)
    print("Overall Sensitivity :",sensitivity)
    print("Overall Specificity :",specificity)
    print("Overall Precision :",precision)
    print("Overall F1_measure :",f1_measure)


    for count in range(0,len(prediction)):

        if prediction[count] == 1 :
            positives = positives + str(sequence_data[count]) + "\n"
        else:
            negatives = negatives + str(sequence_data[count]) + "\n"

    with open("predictions_pos.txt", 'w') as file:
        file.write(positives)

    with open("predictions_neg.txt", 'w') as file:
        file.write(negatives)


