from ensemble import ensemble_pipeline
from aggregate import aggregate_pipeline
import inquirer


print("Welcome to the druggable protein identifier pipeline")

choice = [
    inquirer.List('option',
                message="Select a Classification Method :",
                choices=['Ensemble Model','Aggregated Features']
                )
]

answer = inquirer.prompt(choice)

if str(answer['option']) == "Ensemble Model":
    ensemble_pipeline()
else:
    aggregate_pipeline()
