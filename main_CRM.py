#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 00:40:59 2020

@author: deepthisen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:09:39 2019

@author: deepthisen
"""



import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import RNN
from os.path import join as join
import pandas as pd
import time
from Modules.CRM_module import CRMP as crm



# Specify path to input file 

#filepath = r'Datasets/NonStreak'
#parse_date = False

filepath = r'Datasets/Streak'
parse_date = True #This dataset has dates instead of elapsed time. Hence convert to timedelta

#Path to total reservoir vol injection and production dataset
qi = pd.read_excel(join(filepath,'Injection.xlsx'),index=0)
qp = pd.read_excel(join(filepath,'Production.xlsx'))
percent_train = 0.7




time_colname = 'Time [days]'
if parse_date:
    qi[time_colname]=(qi.Date-qi.Date[0])/pd.to_timedelta(1, unit='D')
    
InjList = [x for x in qi.keys() if x.startswith('I')]
PrdList = [x for x in qp.keys() if x.startswith('P')]
t_arr = qi[time_colname].values


N_inj = len(InjList)
N_prd = len(PrdList)
qi_arr = qi[InjList].values
q_obs = qp[PrdList].values

##Segregating into training and test set
n_train = int(percent_train *len(t_arr))     
q_obs_train = q_obs[:n_train,:]
q_obs_test = q_obs[n_train:,:]
input_series_train = [t_arr[:n_train],qi_arr[:n_train,:]]  
input_series_test = [t_arr[n_train:],qi_arr[n_train:,:]]  

#############################################################################
####################CAPACITANCE-RESISTANCE MODEL#############################
#############################################################################

###Initialization#####
tau = np.ones(N_prd)
gain_mat = np.ones([N_inj,N_prd])
gain_mat = gain_mat/(np.sum(gain_mat,1).reshape([-1,1]))
qp0 =np.array([[0,0,0,0]])
J= np.array([[1,1,1,1]])/10       
inputs_list = [tau,gain_mat,qp0]
crm1 = crm(inputs_list,include_press = False)


###Fitting#####
##initial guess
init_guess = inputs_list
t_start = time.perf_counter()
params_fit = crm1.fit_model(input_series_train,q_obs_train,init_guess)
t_taken = time.perf_counter() - t_start
print('Time taken to fit : '+str(t_taken))
###Prediction######
qp_pred_train = crm1.prod_pred(input_series_train)
qp_pred_test = crm1.prod_pred(input_series_test)


###Print Results.###################
for i in range(N_prd):
    plt.plot(t_arr,q_obs[:,i],linestyle='-',c='r',label = 'Actual')
    plt.plot(t_arr[:n_train],qp_pred_train[:,i],linestyle='-',c='b',label = 'CRM Train')
    plt.plot(t_arr[n_train:],qp_pred_test[:,i],linestyle='--',marker=None,c='b',label = 'CRM Test')
    plt.title('P'+str(i+1),fontsize=15)
    plt.ylabel('Total Fluid (RB)',fontsize=13)
    plt.xlabel('Time (D)',fontsize=13)
    plt.legend(fontsize=11)
    plt.show()
    
### Calculate Error ###############

#CRM
train_err_crm = np.sqrt(np.mean((q_obs_train-qp_pred_train)**2,axis=0))
test_err_crm = np.sqrt(np.mean((q_obs_test-qp_pred_test)**2,axis=0))
