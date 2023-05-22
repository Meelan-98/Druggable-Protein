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


def get_predictor(models,dataset):

    best_model = find_best_model(dataset,models)

    full_train_X = dataset.drop('druggable', axis=1)
    full_train_y = dataset['druggable']

    print("Most Suitable predictor is :",best_model)

    predictor = get_model(best_model)

    predictor.fit(full_train_X, full_train_y)

    return(predictor)

def find_best_model(dataset,models):

    max_acc = 0
    best_model = ""

    for model_name in models:
    
        accuracies = []

        num_folds = 5

        kf = KFold(n_splits=num_folds, shuffle=True)
        fold_indices = kf.split(dataset)

        for fold, (train_indices, test_indices) in enumerate(fold_indices):

            train_data = dataset.iloc[train_indices]
            test_data = dataset.iloc[test_indices]

            train_X = train_data.drop('druggable', axis=1)
            train_y = train_data['druggable']

            test_X = test_data.drop('druggable', axis=1)
            test_y = test_data['druggable']

            model = get_model(model_name)

            model.fit(train_X, train_y)

            y_pred = model.predict(test_X)

            accuracy = accuracy_score(test_y, y_pred)

            accuracies.append(accuracy)

        avg_acc = sum(accuracies) / len(accuracies)
        
        if (avg_acc > max_acc):
            max_acc = avg_acc
            best_model = model_name
        
    return(best_model)

def get_model(model_name):

    if(model_name=="Logistic Regression"):

        model = LogisticRegression(max_iter=500)
    
    elif(model_name=="Support Vector Machines"):

        model = LinearSVC(max_iter=1000)

    elif(model_name=="Decision Trees"):

        model = DecisionTreeClassifier()

    elif(model_name=="Random Forest"):

        model = RandomForestClassifier()
    
    elif(model_name=="Naive Bayes"):

        model = GaussianNB()
    
    elif(model_name=="K-Nearest Neighbor"):

        model = KNeighborsClassifier()

    return(model)

def ensemble_pipeline():
    
    data_names = ["AAC","CTD","PAAC","DPC"]
    models = ["Naive Bayes","Logistic Regression","Random Forest"]

    results = []

    for d_name in data_names:

        dataset = process_data(d_name)
        model = get_predictor(models,dataset[0])

        testData = dataset[1]

        full_test_X = testData.drop(columns=['seq_name', 'druggable'], axis=1)
        full_test_y = testData['druggable']

        results.append(model.predict(full_test_X))

    sequence_data = testData['seq_name'].tolist()
    positives = ""
    negatives = ""

    sum_array = results[0]

    for i in range(1,len(data_names)):
        sum_array = sum_array + results[i]

    combined_array = np.where(sum_array >= 2, 1, 0)

    accuracy = accuracy_score(full_test_y, combined_array)
    sensitivity = recall_score(full_test_y, combined_array)
    specificity = recall_score(full_test_y, combined_array, pos_label=0)
    precision = precision_score(full_test_y, combined_array)
    f1_measure = f1_score(full_test_y, combined_array)

    print("Overall Accuracy :",accuracy)
    print("Overall Sensitivity :",sensitivity)
    print("Overall Specificity :",specificity)
    print("Overall Precision :",precision)
    print("Overall F1_measure :",f1_measure)


    for count in range(0,len(combined_array)):

        if combined_array[count] == 1 :
            positives = positives + str(sequence_data[count]) + "\n"
        else:
            negatives = negatives + str(sequence_data[count]) + "\n"

    with open("predictions_pos.txt", 'w') as file:
        file.write(positives)

    with open("predictions_neg.txt", 'w') as file:
        file.write(negatives)


