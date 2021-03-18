from  MarchMadness2020 import *

import torch
from torch.autograd import Variable
from torch import optim

from UtilityFunctions import predict
optimal_lm = torch.load('optimal_learningmodel.pt')


#W = model.linear.weight
#W = optimal_lm.linear.weight
#F = [F1, F2]
#R1 = sum(W[0] * X)
#R2 = sum(W[1] * X)
#print(R1)
#print(R2)

#Midwest, East, South, West
#teams = [Kansas, PrairieViewA_M, NorthCarolinaA_T, Houston, Virginia, Iowa, MississippiState, Providence, Louisville, Vermont, OhioState, RhodeIsland, Villanova, Hofstra, WestVirginia, EastTenneseeState, Maryland, LittleRock, SanDiegoState, EasternWashington, ArizonaState, TexasTech, Auburn, Cincinnati, PennState, StephenFAustin, Marquette, Rutgers, Duke, BowlingGreen, Michigan, NorthernIowa, Dayton, Colgate, Baylor, Siena, StFrancis, Florida, Indiana, Arizona, Liberty, MichiganState, NorthTexas, Illinois, Stanford, Creighton, WrightState, Butler, UtahState, FloridaState, Radford, Gonzaga, SouthDakota, LSU, USC, Wisconsin, Yale, Oregon, NewMexicoState, BYU, Oklahoma, Xavier, Kentucky, Belmont, Colorado, SaintMarys, SetonHall, UCIrvine]

#getTeam function to find team name
def getTeam(statistic):
    for team, stat in teams.items():
        if statistic == stat:
            return team

#gets the name of all the teams in the round
def getRound(tournament):
    for team in tournament:
        print(getTeam(team))

#create an array of arrays that symbolizes a collection of team stats
def createTournament(dictTeams):
    for teamStat in teams.values():
        tournament.append(teamStat)

#matchup function to use predict an implement between two teams
def matchup(team0, team1):
    game = [team0 + team1]
    game = torch.tensor(game)
    #checks if which teams wins
    if (predict(optimal_lm, game) == 0):
        tournament.remove(team1)
    else:
        tournament.remove(team0)

#simulates one round
def roundSim(currentRound):
    for i in range(0, int(len(currentRound)/2)):
        matchup(currentRound[i], currentRound[i+1])

#simulates through entire 7 rounds
def sim(tournament):
    for round in range(7):
        roundSim(tournament)
    return getTeam(tournament[0])

tournament = []
createTournament(teams)

#first Four simulation
matchup(teams['PrairieViewA_M'], teams['NorthCarolinaA_T'])
matchup(teams['MississippiState'], teams['Providence'])
matchup(teams['Siena'], teams['StFrancis'])
matchup(teams['Oklahoma'], teams['Xavier'])

print("Round of 64")
print(getRound(tournament))
roundSim(tournament)
print("Round of 32")
print(getRound(tournament))
roundSim(tournament)
print("Sweet 16")
print(getRound(tournament))
roundSim(tournament)
print("Elite 8")
print(getRound(tournament))
roundSim(tournament)
print("Final 4")
print(getRound(tournament))
roundSim(tournament)
print("Championship")
print(getRound(tournament))
roundSim(tournament)
print("Champion")
print(getRound(tournament))
roundSim(tournament)





#TESTING TENSOR WEIGHTS:
#W = optimal_lm.linear.weight
#a = teams['SanDiegoState'] + teams['EasternWashington']
#a = [a]
#a = torch.tensor(a)

#print(predict(optimal_lm, a))
#R1 = (W[0] * a)
#R2 = (W[1] * a)
#print(R1)
#print(R2)
#print(torch.sum(W[0] * a))
#print(torch.sum(W[1] * a))
#W = model.linear.weight
#W = optimal_lm.linear.weight
#F = [F1, F2]
#R1 = torch.sum(W[0] * X)
#R2 = torch.sum(W[1] * X)
#print(R1)
#print(R2)

#if you find a pateern, you should i remove it
#[Fseed, Ffgm, Ffga, Ffgm3, Ffga3, Fftm, Ffta, For, Fdr, Fast, Fto, Fstl, Fblk, Fpf,
#problematic datapoints:
