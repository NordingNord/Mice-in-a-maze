import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import pandas as pd
from scipy import stats

def convert_text_to_lists(filename):
    with open(filename, 'r') as file:
        random_scores = np.fromstring(file.readline(), dtype=np.float64, sep=',')
        sarsa_scores = np.fromstring(file.readline(), dtype=np.float64, sep=',')
        steps = np.fromstring(file.readline(), dtype=np.float64, sep=',')
        sarsa_total_wins = np.fromstring(file.readline(), dtype=np.float64, sep=',')
        sarsa_total_score = np.fromstring(file.readline(), dtype=np.float64, sep=',')
        random_total_win = np.fromstring(file.readline(), dtype=np.float64, sep=',')
        random_total_score = np.fromstring(file.readline(), dtype=np.float64, sep=',')
        total_ties = np.fromstring(file.readline(), dtype=np.float64, sep=',')
        episodes = random_scores.size
        return random_scores,sarsa_scores,steps,sarsa_total_wins,sarsa_total_score,random_total_win,random_total_score,total_ties,episodes

def convert_col_to_lists(filename):
    data_file = pd.read_csv(filename,index_col=None)
    a=data_file.iloc[:,0].values
    b=data_file.iloc[:,1].values
    c=data_file.iloc[:,2].values
    d=data_file.iloc[:,3].values
    return a,b,c,d

def line_plot(x,y,labels,x_label,y_label,title,colors):
    plt.figure(figsize=(10,6))
    for index in range(len(y)):
        plt.plot(x,y[index], colors[index], label = labels[index])
    
    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(title+".pdf")
    plt.show()

def line_point_plot(x,y,labels,x_label,y_label,title,colors):
    plt.figure(figsize=(10,6))
    for index in range(len(y)):
        plt.plot(x[index],y[index], colors[index],marker="o",label = labels[index])
        #for i,j in zip(x[index],y[index]):
            #plt.annotate("("+str(i)+","+str(j)+")",xy=(i,j),xytext=(5,4), textcoords='offset points')
    
    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(title+".pdf")
    plt.show()

def bar_plot(names,values,x_label,y_label,title):
    start_index = 0
    plt.figure(figsize=(11,6))
    for index in range(len(names)):
        plt.bar(names[index],values[index])
        for i in range(len(values[index])):
            plt.text(start_index+i, values[index][i], values[index][i], ha = 'center')
        start_index += len(values[index])
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(title+".pdf")
    plt.show()

# get epsilon data  
#epsilon_0_0_5_random_scores,epsilon_0_0_5_sarsa_scores,epsilon_0_0_5_steps,epsilon_0_0_5_sarsa_total_wins,epsilon_0_0_5_sarsa_total_score,epsilon_0_0_5_random_total_win,epsilon_0_0_5_random_total_score,epsilon_0_0_5_total_ties,epsilon_0_0_5_episodes =convert_text_to_lists("Sarsa_versus_random_0_0_5.csv")
#epsilon_0_1_random_scores,epsilon_0_1_sarsa_scores,epsilon_0_1_steps,epsilon_0_1_sarsa_total_wins,epsilon_0_1_sarsa_total_score,epsilon_0_1_random_total_win,epsilon_0_1_random_total_score,epsilon_0_1_total_ties,epsilon_0_1_episodes =convert_text_to_lists("Sarsa_versus_random_0_1.csv")
#epsilon_0_2_random_scores,epsilon_0_2_sarsa_scores,epsilon_0_2_steps,epsilon_0_2_sarsa_total_wins,epsilon_0_2_sarsa_total_score,epsilon_0_2_random_total_win,epsilon_0_2_random_total_score,epsilon_0_2_total_ties,epsilon_0_2_episodes =convert_text_to_lists("Sarsa_versus_random_0_2.csv")

# get learning data
#learning_0_2_random_scores,learning_0_2_sarsa_scores,learning_0_2_steps,learning_0_2_sarsa_total_wins,learning_0_2_sarsa_total_score,learning_0_2_random_total_win,learning_0_2_random_total_score,learning_0_2_total_ties,learning_0_2_episodes =convert_text_to_lists("Sarsa_versus_learning_0_2.csv")
#learning_0_5_random_scores,learning_0_5_sarsa_scores,learning_0_5_steps,learning_0_5_sarsa_total_wins,learning_0_5_sarsa_total_score,learning_0_5_random_total_win,learning_0_5_random_total_score,learning_0_5_total_ties,learning_0_5_episodes =convert_text_to_lists("Sarsa_versus_learning_0_5.csv")
#learning_0_8_random_scores,learning_0_8_sarsa_scores,learning_0_8_steps,learning_0_8_sarsa_total_wins,learning_0_8_sarsa_total_score,learning_0_8_random_total_win,learning_0_8_random_total_score,learning_0_8_total_ties,learning_0_8_episodes =convert_text_to_lists("Sarsa_versus_learning_0_8.csv")

# get gamma data
#gamma_0_2_random_scores,gamma_0_2_sarsa_scores,gamma_0_2_steps,gamma_0_2_sarsa_total_wins,gamma_0_2_sarsa_total_score,gamma_0_2_random_total_win,gamma_0_2_random_total_score,gamma_0_2_total_ties,gamma_0_2_episodes =convert_text_to_lists("Sarsa_versus_gamma_0_2.csv")
#gamma_0_5_random_scores,gamma_0_5_sarsa_scores,gamma_0_5_steps,gamma_0_5_sarsa_total_wins,gamma_0_5_sarsa_total_score,gamma_0_5_random_total_win,gamma_0_5_random_total_score,gamma_0_5_total_ties,gamma_0_5_episodes =convert_text_to_lists("Sarsa_versus_gamma_0_5.csv")
#gamma_0_8_random_scores,gamma_0_8_sarsa_scores,gamma_0_8_steps,gamma_0_8_sarsa_total_wins,gamma_0_8_sarsa_total_score,gamma_0_8_random_total_win,gamma_0_8_random_total_score,gamma_0_8_total_ties,gamma_0_8_episodes =convert_text_to_lists("Sarsa_versus_gamma_0_8.csv")

# get episode results
#episodes_100_random_scores,episodes_100_sarsa_scores,episodes_100_steps,episodes_100_sarsa_total_wins,episodes_100_sarsa_total_score,episodes_100_random_total_win,episodes_100_random_total_score,episodes_100_total_ties,episodes_100_episodes =convert_text_to_lists("Sarsa_versus_random_100_episodes.csv")
#episodes_500_random_scores,episodes_500_sarsa_scores,episodes_500_steps,episodes_500_sarsa_total_wins,episodes_500_sarsa_total_score,episodes_500_random_total_win,episodes_500_random_total_score,episodes_500_total_ties,episodes_500_episodes =convert_text_to_lists("Sarsa_versus_random_500_episodes.csv")
#episodes_1000_random_scores,episodes_1000_sarsa_scores,episodes_1000_steps,episodes_1000_sarsa_total_wins,episodes_1000_sarsa_total_score,episodes_1000_random_total_win,episodes_1000_random_total_score,episodes_1000_total_ties,episodes_1000_episodes =convert_text_to_lists("Sarsa_versus_random_1000_episodes.csv")
#episodes_1500_random_scores,episodes_1500_sarsa_scores,episodes_1500_steps,episodes_1500_sarsa_total_wins,episodes_1500_sarsa_total_score,episodes_1500_random_total_win,episodes_1500_random_total_score,episodes_1500_total_ties,episodes_1500_episodes =convert_text_to_lists("Sarsa_versus_random_1500_episodes.csv")
#episodes_2000_random_scores,episodes_2000_sarsa_scores,episodes_2000_steps,episodes_2000_sarsa_total_wins,episodes_2000_sarsa_total_score,episodes_2000_random_total_win,episodes_2000_random_total_score,episodes_2000_total_ties,episodes_2000_episodes =convert_text_to_lists("Sarsa_versus_random_2000_episodes.csv")

# plot epsilon results
#means_epsilon = [np.mean(epsilon_0_0_5_sarsa_scores),np.mean(epsilon_0_1_sarsa_scores),np.mean(epsilon_0_2_sarsa_scores)]
#line_point_plot([[0.05,0.1,0.2]],[means_epsilon],["mean score"],"epsilon value", "amount of cheese eaten [score]","Epsilon SARSA results",["b"])

# plot learning results
#means_alpha = [np.mean(learning_0_2_sarsa_scores),np.mean(learning_0_5_sarsa_scores),np.mean(learning_0_8_sarsa_scores)]
#line_point_plot([[0.2,0.5,0.8]],[means_alpha],["mean score"],"alpha value", "amount of cheese eaten [score]","Alpha SARSA results",["b"])
#line_plot(np.arange(0,learning_0_2_episodes,1, dtype=int),[learning_0_2_sarsa_scores,learning_0_5_sarsa_scores,learning_0_8_sarsa_scores],["alpha 0.2","alpha 0.5","alpha 0.8"],"episodes", "amount of cheese eaten [score]","Alpha SARSA results",["b","g","r"])

# plot gamma results
#means_gamma = [np.mean(gamma_0_2_sarsa_scores),np.mean(gamma_0_5_sarsa_scores),np.mean(gamma_0_8_sarsa_scores)]
#line_point_plot([[0.2,0.5,0.8]],[means_gamma],["mean score"],"gamma value", "amount of cheese eaten [score]","Gamma SARSA results",["b"])
#line_plot(np.arange(0,gamma_0_2_episodes,1, dtype=int),[gamma_0_2_sarsa_scores,gamma_0_5_sarsa_scores,gamma_0_8_sarsa_scores],["gamma 0.2","gamma 0.5","gamma 0.8"],"episodes", "amount of cheese eaten [score]","Alpha SARSA results",["b","g","r"])
# combined
#line_point_plot([[0.05,0.1,0.2],[0.2,0.5,0.8],[0.2,0.5,0.8]],[means_epsilon,means_alpha,means_gamma],["Epsilon","Alpha","Gamma"],"Value of parameter", "Mean amount of cheese eaten [mean score]","SARSA mean score for each parameter setting",["b","g","r"])


# plot total scores
#bar_plot([["epsilon 0.05", "epsilon 0.1", "epsilon 0.2"],["alpha 0.2", "alpha 0.5", "alpha 0.8"],["gamma 0.2", "gamma 0.5", "gamma 0.8"]],[[epsilon_0_0_5_sarsa_total_score[0],epsilon_0_1_sarsa_total_score[0],epsilon_0_2_sarsa_total_score[0]],[learning_0_2_sarsa_total_score[0],learning_0_5_sarsa_total_score[0],learning_0_8_sarsa_total_score[0]],[gamma_0_2_sarsa_total_score[0],gamma_0_5_sarsa_total_score[0],gamma_0_8_sarsa_total_score[0]]],"Setup", "Total amount of cheese eaten [score]","Total scores SARSA bar")

# plot total wins
#bar_plot([["epsilon 0.05", "epsilon 0.1", "epsilon 0.2"],["alpha 0.2", "alpha 0.5", "alpha 0.8"],["gamma 0.2", "gamma 0.5", "gamma 0.8"]],[[epsilon_0_0_5_sarsa_total_wins[0],epsilon_0_1_sarsa_total_wins[0],epsilon_0_2_sarsa_total_wins[0]],[learning_0_2_sarsa_total_wins[0],learning_0_5_sarsa_total_wins[0],learning_0_8_sarsa_total_wins[0]],[gamma_0_2_sarsa_total_wins[0],gamma_0_5_sarsa_total_wins[0],gamma_0_8_sarsa_total_wins[0]]],"Parameter settings", "Total wins over random","Total wins SARSA barplot")

# plot total wins per episode training
#line_point_plot([[100,500,1000,1500,2000]],[[episodes_100_sarsa_total_wins[0],episodes_500_sarsa_total_wins[0],episodes_1000_sarsa_total_wins[0],episodes_1500_sarsa_total_wins[0],episodes_2000_sarsa_total_wins[0]]],["Wins out of 5000"],"Number of Episodes used during training", "Total wins over random mouse","SARSA total wins over random mouse",["b"])
#bar_plot([["100 episodes", "500 episodes", "1000 episodes", "1500 episodes", "2000 episodes"]],[[episodes_100_sarsa_total_wins[0],episodes_500_sarsa_total_wins[0],episodes_1000_sarsa_total_wins[0],episodes_1500_sarsa_total_wins[0],episodes_2000_sarsa_total_wins[0]]],"Episodes used during training", "Total wins over random","Total wins SARSA per episode")

# plot total score per episode training
#line_point_plot([[100,500,1000,1500,2000]],[[episodes_100_sarsa_total_score[0],episodes_500_sarsa_total_score[0],episodes_1000_sarsa_total_score[0],episodes_1500_sarsa_total_score[0],episodes_2000_sarsa_total_score[0]]],["Total score"],"episodes used during training", "Total score during 5000 episodes","Total scores SARSA results per episode",["b"])
#bar_plot([["100 episodes", "500 episodes", "1000 episodes", "1500 episodes", "2000 episodes"]],[[episodes_100_sarsa_total_score[0],episodes_500_sarsa_total_score[0],episodes_1000_sarsa_total_score[0],episodes_1500_sarsa_total_score[0],episodes_2000_sarsa_total_score[0]]],"Episodes used during training", "Total amount of cheese eaten (score)","Total score SARSA per episode")

# plot steps episode training
#means = [np.mean(episodes_100_steps),np.mean(episodes_500_steps),np.mean(episodes_1000_steps),np.mean(episodes_1500_steps),np.mean(episodes_2000_steps)]
#line_point_plot([[100,500,1000,1500,2000]],[means],["Mean steps taken"],"Episodes used during training", "Mean steps taken for all cheese to be eaten","Steps taken for each episode to end",["b"])
#line_plot(np.arange(0,episodes_100_episodes,1, dtype=int),[episodes_100_steps,episodes_500_steps,episodes_1000_steps,episodes_1500_steps,episodes_2000_steps],["100 training episodes","500 training episodes","1000 training episodes","1500 training episodes","2000 training episodes"],"episodes", "Steps taken","Steps taken for different training episodes",["b","g","r","y","c"])

# get moving win graph
#sarsa_win,random_win,tie = convert_col_to_lists("Moving_wins.csv")
#line_plot(np.arange(0,episodes_100_episodes,1, dtype=int),[sarsa_win,random_win,tie],["SARSA mouse","Random mouse","Ties"],"Episodes", "Current number of wins","Wins during 5000 episodes",["b","g","r"])





# get epsilon data  
#epsilon_0_0_5_random_scores,epsilon_0_0_5_expected_sarsa_scores,epsilon_0_0_5_steps,epsilon_0_0_5_expected_sarsa_total_wins,epsilon_0_0_5_expected_sarsa_total_score,epsilon_0_0_5_random_total_win,epsilon_0_0_5_random_total_score,epsilon_0_0_5_total_ties,epsilon_0_0_5_episodes =convert_text_to_lists("Expected sarsa_versus_random_0_0_5_epsilon.csv")
#epsilon_0_1_random_scores,epsilon_0_1_expected_sarsa_scores,epsilon_0_1_steps,epsilon_0_1_expected_sarsa_total_wins,epsilon_0_1_expected_sarsa_total_score,epsilon_0_1_random_total_win,epsilon_0_1_random_total_score,epsilon_0_1_total_ties,epsilon_0_1_episodes =convert_text_to_lists("Expected sarsa_versus_random_0_1_epsilon.csv")
#epsilon_0_2_random_scores,epsilon_0_2_expected_sarsa_scores,epsilon_0_2_steps,epsilon_0_2_expected_sarsa_total_wins,epsilon_0_2_expected_sarsa_total_score,epsilon_0_2_random_total_win,epsilon_0_2_random_total_score,epsilon_0_2_total_ties,epsilon_0_2_episodes =convert_text_to_lists("Expected sarsa_versus_random_0_2_epsilon.csv")
# get alpha data
#alpha_0_2_random_scores,alpha_0_2_expected_sarsa_scores,alpha_0_2_steps,alpha_0_2_expected_sarsa_total_wins,alpha_0_2_expected_sarsa_total_score,alpha_0_2_random_total_win,alpha_0_2_random_total_score,alpha_0_2_total_ties,alpha_0_2_episodes =convert_text_to_lists("Expected sarsa_versus_random_0_2_alpha.csv")
#alpha_0_5_random_scores,alpha_0_5_expected_sarsa_scores,alpha_0_5_steps,alpha_0_5_expected_sarsa_total_wins,alpha_0_5_expected_sarsa_total_score,alpha_0_5_random_total_win,alpha_0_5_random_total_score,alpha_0_5_total_ties,alpha_0_5_episodes =convert_text_to_lists("Expected sarsa_versus_random_0_5_alpha.csv")
#alpha_0_8_random_scores,alpha_0_8_expected_sarsa_scores,alpha_0_8_steps,alpha_0_8_expected_sarsa_total_wins,alpha_0_8_expected_sarsa_total_score,alpha_0_8_random_total_win,alpha_0_8_random_total_score,alpha_0_8_total_ties,alpha_0_8_episodes =convert_text_to_lists("Expected sarsa_versus_random_0_8_alpha.csv")
# get gamma data
#gamma_0_2_random_scores,gamma_0_2_expected_sarsa_scores,gamma_0_2_steps,gamma_0_2_expected_sarsa_total_wins,gamma_0_2_expected_sarsa_total_score,gamma_0_2_random_total_win,gamma_0_2_random_total_score,gamma_0_2_total_ties,gamma_0_2_episodes =convert_text_to_lists("Expected sarsa_versus_random_0_2_gamma.csv")
#gamma_0_5_random_scores,gamma_0_5_expected_sarsa_scores,gamma_0_5_steps,gamma_0_5_expected_sarsa_total_wins,gamma_0_5_expected_sarsa_total_score,gamma_0_5_random_total_win,gamma_0_5_random_total_score,gamma_0_5_total_ties,gamma_0_5_episodes =convert_text_to_lists("Expected sarsa_versus_random_0_5_gamma.csv")
#gamma_0_8_random_scores,gamma_0_8_expected_sarsa_scores,gamma_0_8_steps,gamma_0_8_expected_sarsa_total_wins,gamma_0_8_expected_sarsa_total_score,gamma_0_8_random_total_win,gamma_0_8_random_total_score,gamma_0_8_total_ties,gamma_0_8_episodes =convert_text_to_lists("Expected sarsa_versus_random_0_8_gamma.csv")
# get episode results
episodes_100_random_scores,episodes_100_expected_sarsa_scores,episodes_100_steps,episodes_100_expected_sarsa_total_wins,episodes_100_expected_sarsa_total_score,episodes_100_random_total_win,episodes_100_random_total_score,episodes_100_total_ties,episodes_100_episodes =convert_text_to_lists("Expected sarsa_versus_random_100_episodes.csv")
#episodes_500_random_scores,episodes_500_expected_sarsa_scores,episodes_500_steps,episodes_500_expected_sarsa_total_wins,episodes_500_expected_sarsa_total_score,episodes_500_random_total_win,episodes_500_random_total_score,episodes_500_total_ties,episodes_500_episodes =convert_text_to_lists("Expected sarsa_versus_random_500_episodes.csv")
#episodes_1000_random_scores,episodes_1000_expected_sarsa_scores,episodes_1000_steps,episodes_1000_expected_sarsa_total_wins,episodes_1000_expected_sarsa_total_score,episodes_1000_random_total_win,episodes_1000_random_total_score,episodes_1000_total_ties,episodes_1000_episodes =convert_text_to_lists("Expected sarsa_versus_random_1000_episodes.csv")
#episodes_1500_random_scores,episodes_1500_expected_sarsa_scores,episodes_1500_steps,episodes_1500_expected_sarsa_total_wins,episodes_1500_expected_sarsa_total_score,episodes_1500_random_total_win,episodes_1500_random_total_score,episodes_1500_total_ties,episodes_1500_episodes =convert_text_to_lists("Expected sarsa_versus_random_1500_episodes.csv")
#episodes_2000_random_scores,episodes_2000_expected_sarsa_scores,episodes_2000_steps,episodes_2000_expected_sarsa_total_wins,episodes_2000_expected_sarsa_total_score,episodes_2000_random_total_win,episodes_2000_random_total_score,episodes_2000_total_ties,episodes_2000_episodes =convert_text_to_lists("Expected sarsa_versus_random_2000_episodes.csv")

# plot epsilon results
#means_epsilon = [np.mean(epsilon_0_0_5_expected_sarsa_scores),np.mean(epsilon_0_1_expected_sarsa_scores),np.mean(epsilon_0_2_expected_sarsa_scores)]
#line_point_plot([[0.05,0.1,0.2]],[means_epsilon],["mean score"],"epsilon value", "amount of cheese eaten [score]","Epsilon SARSA results",["b"])

# plot learning results
#means_alpha = [np.mean(alpha_0_2_expected_sarsa_scores),np.mean(alpha_0_5_expected_sarsa_scores),np.mean(alpha_0_8_expected_sarsa_scores)]
#line_point_plot([[0.2,0.5,0.8]],[means_alpha],["mean score"],"alpha value", "amount of cheese eaten [score]","Alpha SARSA results",["b"])
#line_plot(np.arange(0,learning_0_2_episodes,1, dtype=int),[learning_0_2_sarsa_scores,learning_0_5_sarsa_scores,learning_0_8_sarsa_scores],["alpha 0.2","alpha 0.5","alpha 0.8"],"episodes", "amount of cheese eaten [score]","Alpha SARSA results",["b","g","r"])

# plot gamma results
#means_gamma = [np.mean(gamma_0_2_expected_sarsa_scores),np.mean(gamma_0_5_expected_sarsa_scores),np.mean(gamma_0_8_expected_sarsa_scores)]
#line_point_plot([[0.2,0.5,0.8]],[means_gamma],["mean score"],"gamma value", "amount of cheese eaten [score]","Gamma SARSA results",["b"])
#line_plot(np.arange(0,gamma_0_2_episodes,1, dtype=int),[gamma_0_2_sarsa_scores,gamma_0_5_sarsa_scores,gamma_0_8_sarsa_scores],["gamma 0.2","gamma 0.5","gamma 0.8"],"episodes", "amount of cheese eaten [score]","Alpha SARSA results",["b","g","r"])
# combined
#line_point_plot([[0.05,0.1,0.2],[0.2,0.5,0.8],[0.2,0.5,0.8]],[means_epsilon,means_alpha,means_gamma],["Epsilon","Alpha","Gamma"],"Value of parameter", "Mean amount of cheese eaten [mean score]","Expected SARSA mean score for each parameter setting",["b","g","r"])


# plot total scores
#bar_plot([["epsilon 0.05", "epsilon 0.1", "epsilon 0.2"],["alpha 0.2", "alpha 0.5", "alpha 0.8"],["gamma 0.2", "gamma 0.5", "gamma 0.8"]],[[epsilon_0_0_5_sarsa_total_score[0],epsilon_0_1_sarsa_total_score[0],epsilon_0_2_sarsa_total_score[0]],[learning_0_2_sarsa_total_score[0],learning_0_5_sarsa_total_score[0],learning_0_8_sarsa_total_score[0]],[gamma_0_2_sarsa_total_score[0],gamma_0_5_sarsa_total_score[0],gamma_0_8_sarsa_total_score[0]]],"Setup", "Total amount of cheese eaten [score]","Total scores SARSA bar")

# plot total wins
#bar_plot([["epsilon 0.05", "epsilon 0.1", "epsilon 0.2"],["alpha 0.2", "alpha 0.5", "alpha 0.8"],["gamma 0.2", "gamma 0.5", "gamma 0.8"]],[[epsilon_0_0_5_expected_sarsa_total_wins[0],epsilon_0_1_expected_sarsa_total_wins[0],epsilon_0_2_expected_sarsa_total_wins[0]],[alpha_0_2_expected_sarsa_total_wins[0],alpha_0_5_expected_sarsa_total_wins[0],alpha_0_8_expected_sarsa_total_wins[0]],[gamma_0_2_expected_sarsa_total_wins[0],gamma_0_5_expected_sarsa_total_wins[0],gamma_0_8_expected_sarsa_total_wins[0]]],"Parameter settings", "Total wins over random","Total wins Expected SARSA barplot")

# plot total wins per episode training
#line_point_plot([[100,500,1000,1500,2000]],[[episodes_100_expected_sarsa_total_wins[0],episodes_500_expected_sarsa_total_wins[0],episodes_1000_expected_sarsa_total_wins[0],episodes_1500_expected_sarsa_total_wins[0],episodes_2000_expected_sarsa_total_wins[0]]],["Wins out of 5000"],"Number of Episodes used during training", "Total wins over random mouse","Expected SARSA total wins over random mouse",["b"])

# plot steps episode training
#means = [np.mean(episodes_100_steps),np.mean(episodes_500_steps),np.mean(episodes_1000_steps),np.mean(episodes_1500_steps),np.mean(episodes_2000_steps)]
#line_point_plot([[100,500,1000,1500,2000]],[means],["Mean steps taken"],"Episodes used during training", "Mean steps taken for all cheese to be eaten","Expected SARSA steps taken for each episode to end",["b"])

# get moving win graph
#expected_sarsa_win,random_win,tie = convert_col_to_lists("Moving_wins_expected.csv")
#line_plot(np.arange(0,episodes_100_episodes,1, dtype=int),[expected_sarsa_win,random_win,tie],["Expected SARSA mouse","Random mouse","Ties"],"Episodes", "Current number of wins","Expected SARSA wins during 5000 episodes",["b","g","r"])

# get moving win graph
#expected_sarsa_win,sarsa_win,random_win,tie = convert_col_to_lists("Moving_wins_comparison.csv")
#line_plot(np.arange(0,episodes_100_episodes,1, dtype=int),[expected_sarsa_win,sarsa_win,random_win,tie],["Expected SARSA mouse","SARSA mouse","Random mouse","Ties"],"Episodes", "Current number of wins","Comparison of wins during 5000 episodes",["b","g","r","c"])

# Chi squared test based on win rate between Sarsa and Expected Sarsa
Extended_Sarsa_wins = [1]*3160
Extended_Sarsa_losses = [0]*(5000-3160)
Extended_Sarsa_total = Extended_Sarsa_wins+Extended_Sarsa_losses

Sarsa_wins = [1]*1421
Sarsa_losses = [0]*(5000-1421)
Sarsa_total = Sarsa_wins+Sarsa_losses
print((3160+1421)/2)
seen = [3160,1421]
expected = [(3160+1421)/2,(3160+1421)/2]

chi2,p = stats.chisquare(seen,expected)
print("Chi-squared test statistic:",chi2)
print("p-value",p)


