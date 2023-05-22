from ensemble import ensemble_pipeline
from aggregate import aggregate_pipeline
import inquirer
import sys
from csv import writer
from propy import PyPro
import csv
import pandas as pd

def extract_data():
    print("Data files creating.....")
    #Ask for the four input file names <positive training data> <negative training data> <positive testing data> <negative testing data>
    #Read the data from them and create the dataset folder
    for i in range(1,5):
        path = './' + sys.argv[i] + '.txt'
        spider_pos_file = open(path, "r")
        num_of_lines = len(spider_pos_file.readlines())
        for j in range(0,4):
            if (i==1 and j==0):
                    file_path = "./descriptors/CTD_TR_neg_SPIDER.csv"
            if (i==1 and j==1):
                    file_path = "./descriptors/AAC_TR_neg_SPIDER.csv"
            if (i==1 and j==2):
                    file_path = "./descriptors/PAAC_TR_neg_SPIDER.csv"
            if (i==1 and j==3):
                    file_path = "./descriptors/DPC_TR_neg_SPIDER.csv"
            if (i==1 and j==4):
                    file_path = "./descriptors/APAAC_TR_neg_SPIDER.csv"
            if (i==2 and j==0):
                    file_path = "./descriptors/CTD_TR_pos_SPIDER.csv"
            if (i==2 and j==1):
                    file_path = "./descriptors/AAC_TR_pos_SPIDER.csv"
            if (i==2 and j==2):
                    file_path = "./descriptors/PAAC_TR_pos_SPIDER.csv"
            if (i==2 and j==3):
                    file_path = "./descriptors/DPC_TR_pos_SPIDER.csv"
            if (i==2 and j==4):
                    file_path = "./descriptors/APAAC_TR_pos_SPIDER.csv"
            if (i==3 and j==0):
                    file_path = "./descriptors/CTD_TS_neg_SPIDER.csv"
            if (i==3 and j==1):
                    file_path = "./descriptors/AAC_TS_neg_SPIDER.csv"
            if (i==3 and j==2):
                    file_path = "./descriptors/PAAC_TS_neg_SPIDER.csv"
            if (i==3 and j==3):
                    file_path = "./descriptors/DPC_TS_neg_SPIDER.csv"
            if (i==3 and j==4):
                    file_path = "./descriptors/APAAC_TS_neg_SPIDER.csv"
            if (i==4 and j==0):
                    file_path = "./descriptors/CTD_TS_pos_SPIDER.csv"
            if (i==4 and j==1):
                    file_path = "./descriptors/AAC_TS_pos_SPIDER.csv"
            if (i==4 and j==2):
                    file_path = "./descriptors/PAAC_TS_pos_SPIDER.csv"
            if (i==4 and j==3):
                    file_path = "./descriptors/DPC_TS_pos_SPIDER.csv"
            if (i==4 and j==4):
                    file_path = "./descriptors/APAAC_TS_pos_SPIDER.csv"
            lines= 0      
            while lines<num_of_lines :
                li = [] 
                name = spider_pos_file.readline().strip()
                sequence = spider_pos_file.readline().strip()
                print(sequence)
                li.append(name[1:])         
                with open(file_path, 'a', newline='') as f_object:
                    # Pass this file object to csv.writer()
                    # and get a writer object
                    # writer_object = writer(f_object)
                    writer_object = csv.writer(f_object)

                    DesObject = PyPro.GetProDes(sequence)  # construct a GetProDes object

                    # calculate 30 pseudo amino acid composition descriptors
                    if(j==0):
                        paac = DesObject.GetCTD()

                    # Amino acid compositon descriptors (20).
                    if(j==1):
                        paac = DesObject.GetAAComp()

                    # Type I Pseudo amino acid composition descriptors (default is 30)
                    if(j==2):
                        paac = DesObject.GetPAAC(lamda=10, weight=0.05)

                    # Dipeptide composition descriptors (400).
                    if(j==3):
                        paac = DesObject.GetDPComp()

                    # Amphiphilic (Type II) Pseudo amino acid composition descriptors.
                    if(j==4):
                        paac = DesObject.GetAPAAC(lamda=6, weight=0.5)

                    if lines == 0:
                        attributes = list(paac.keys())
                        attributes.insert(0, "seq_name")
                        attributes.append("druggable")
                        writer_object.writerow(attributes)

                    for t in paac.items():
                        li.append(t[1])
                    
                    #Change to 1 or 0
                    if(i==1 or i==3):
                        li.append("0")
                    writer_object.writerow(li)
                
                    # Close the file object
                    f_object.close()

                lines=lines+2
    print("Data files created successfully")


print("Welcome to the druggable protein identifier pipeline")

extract_data()


choice = [
    inquirer.List('option',
                message="Select a Classification Method ",
                choices=['Ensemble Model','Aggregated Features']
                )
]

answer = inquirer.prompt(choice)

if str(answer['option']) == "Ensemble Model":
    ensemble_pipeline()
else:
    aggregate_pipeline()
