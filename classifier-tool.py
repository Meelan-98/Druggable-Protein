from ensemble import ensemble_pipeline
from aggregate import aggregate_pipeline
import inquirer

def extract_data():
    #Ask for the four input file names <positive training data> <negative training data> <positive testing data> <negative testing data>
    #Read the data from them and create the dataset folder
    print("Extracting")


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
