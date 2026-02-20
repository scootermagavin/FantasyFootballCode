#Vegas Odds - Predicting NFL Outcomes

#how to handle virtual env
#if in brand new project folder type "python3 -m venv {name of virtual env here}"
#Create requirements.txt by typing "python -m pip freeze > requirements.txt"
#source venv/bin/activate to activate VENV
#if requirements.txt exists already type "python -m pip install -r requirements.txt" to install all dependency packages
#type "deactivate" to close down virtual environment

#Metrics Definitions:

#Avg Amounts of Points a Team Scores
#Avg Amount of Points a team gives up
#Total # of yards gained and possession time
#3rd and 4th down conversion
#Number of turnovers

from pydfs_lineup_optimizer import get_optimizer, Site, Sport


optimizer = get_optimizer(Site.DRAFTKINGS, Sport.FOOTBALL)

optimizer.load_players_from_csv("path_to_csv")
