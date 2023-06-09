import pandas as pd
from sklearn.model_selection import KFold,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def process_data():

    DATA_PATH = "dataset/"

    train_neg_ctd = pd.read_csv(DATA_PATH + "CTD" + "_TR_neg_SPIDER.csv")
    train_pos_ctd = pd.read_csv(DATA_PATH + "CTD" + "_TR_pos_SPIDER.csv")
    test_neg_ctd = pd.read_csv(DATA_PATH + "CTD" + "_TS_neg_SPIDER.csv")
    test_pos_ctd = pd.read_csv(DATA_PATH + "CTD" + "_TS_pos_SPIDER.csv")
    train_neg_paac = pd.read_csv(DATA_PATH + "PAAC" + "_TR_neg_SPIDER.csv")
    train_pos_paac = pd.read_csv(DATA_PATH + "PAAC" + "_TR_pos_SPIDER.csv")
    test_neg_paac = pd.read_csv(DATA_PATH + "PAAC" + "_TS_neg_SPIDER.csv")
    test_pos_paac = pd.read_csv(DATA_PATH + "PAAC" + "_TS_pos_SPIDER.csv")
    train_neg_aac = pd.read_csv(DATA_PATH + "AAC" + "_TR_neg_SPIDER.csv")
    train_pos_aac = pd.read_csv(DATA_PATH + "AAC" + "_TR_pos_SPIDER.csv")
    test_neg_aac = pd.read_csv(DATA_PATH + "AAC" + "_TS_neg_SPIDER.csv")
    test_pos_aac = pd.read_csv(DATA_PATH + "AAC" + "_TS_pos_SPIDER.csv")
    train_neg_apaac = pd.read_csv(DATA_PATH + "APAAC" + "_TR_neg_SPIDER.csv")
    train_pos_apaac = pd.read_csv(DATA_PATH + "APAAC" + "_TR_pos_SPIDER.csv")
    test_neg_apaac = pd.read_csv(DATA_PATH + "APAAC" + "_TS_neg_SPIDER.csv")
    test_pos_apaac = pd.read_csv(DATA_PATH + "APAAC" + "_TS_pos_SPIDER.csv")

    train_pos_temp = pd.merge(train_pos_ctd,train_pos_paac,on=["seq_name","druggable"])
    test_pos_temp = pd.merge(test_pos_ctd,test_pos_paac,on=["seq_name","druggable"])
    train_pos_temp2 = pd.merge(train_pos_temp,train_pos_aac,on=["seq_name","druggable"])
    test_pos_temp2 = pd.merge(test_pos_temp,test_pos_aac,on=["seq_name","druggable"])
    train_pos = pd.merge(train_pos_temp2,train_pos_apaac,on=["seq_name","druggable"])
    test_pos = pd.merge(test_pos_temp2,test_pos_apaac,on=["seq_name","druggable"])
    
    train_neg_temp = pd.merge(train_neg_ctd,train_neg_paac,on=["seq_name","druggable"])
    test_neg_temp = pd.merge(test_neg_ctd,test_neg_paac,on=["seq_name","druggable"])
    train_neg_temp2 = pd.merge(train_neg_temp,train_neg_aac,on=["seq_name","druggable"])
    test_neg_temp2 = pd.merge(test_neg_temp,test_neg_aac,on=["seq_name","druggable"])
    train_neg = pd.merge(train_neg_temp2,train_neg_apaac,on=["seq_name","druggable"])
    test_neg = pd.merge(test_neg_temp2,test_neg_apaac,on=["seq_name","druggable"])

    train_frames = [train_neg, train_pos]
    test_frames = [test_neg, test_pos]

    train_df= pd.concat(train_frames)
    test_df = pd.concat(test_frames)

    columns = list(train_df.columns)
    columns.remove("druggable")
    columns.remove("seq_name")

    train_set = train_df.drop(['seq_name'] , axis=1)
    test_set = test_df.drop(['seq_name'] , axis=1)

    train_final = train_set.copy()
    test_final = test_set.copy()

    for column in columns:
        mean = train_set[column].mean()
        deviation = train_set[column].std()

        train_final[column] = (train_set[column] - mean) / deviation
        test_final[column] = (test_final[column] - mean) / deviation

    train_features = train_final.drop('druggable', axis=1)  
    train_target = train_final['druggable']

    test_features = test_final.drop('druggable', axis=1)  
    test_target = test_final['druggable']

    n_components = 50

    pca = PCA(n_components) 

    pca_train = pca.fit_transform(train_features)
    pca_test = pca.transform(test_features)

    return([[pca_train,train_target],[pca_test,test_target],[test_df['seq_name'].tolist(),0]])

def get_predictor(models,dataset):

    best_model = find_best_model(dataset,models)

    full_train_X = dataset[0]
    full_train_y = dataset[1]

    print("Most Suitable predictor is :",best_model)

    predictor = get_model(best_model)

    predictor.fit(full_train_X, full_train_y)

    return(predictor)

def find_best_model(dataset,models):

    max_acc = 0
    best_model = ""

    for model_name in models:

        model = get_model(model_name)

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        scores = cross_val_score(model, dataset[0], dataset[1], cv=kfold)

        mean_accuracy = np.mean(scores)
        
        if (mean_accuracy > max_acc):
            max_acc = mean_accuracy
            best_model = model_name
        
    return(best_model)

def get_model(model_name):

    if(model_name=="Logistic Regression"):

        model = LogisticRegression(max_iter=500)
    
    elif(model_name=="Support Vector Machines"):

        model = LinearSVC(max_iter=1500)

    elif(model_name=="Decision Trees"):

        model = DecisionTreeClassifier()

    elif(model_name=="Random Forest"):

        model = RandomForestClassifier()
    
    elif(model_name=="Naive Bayes"):

        model = GaussianNB()
    
    elif(model_name=="K-Nearest Neighbor"):

        model = KNeighborsClassifier()

    return(model)

def aggregate_pipeline():

    models = ["Support Vector Machines","Naive Bayes","Logistic Regression","Random Forest","K-Nearest Neighbor","Decision Trees"]
    
    dataset = process_data()
    model = get_predictor(models,dataset[0])

    testData = dataset[1]

    full_test_X = testData[0]
    full_test_y = testData[1]

    prediction = model.predict(full_test_X)

    sequence_data = dataset[2][0]
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



