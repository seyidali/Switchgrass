# -*- coding: utf-8 -*-
"""
Created on Monday 3 February 2025

@author: Seyid Amjad Ali
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import explained_variance_score, max_error,mean_squared_log_error, median_absolute_error, mean_poisson_deviance, mean_gamma_deviance
from xgboost import XGBRegressor
# preprocessing
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, QuantileTransformer, PowerTransformer, Binarizer

input_filename = 'Switchgrass_Data.xlsx'
output_filename = 'output_XGBoost_SwitchGrass(AI).txt'

data_orj = pd.read_excel(input_filename)

# Preprocessing
stdsc = StandardScaler()

data_pp = data_orj

# AI -- 2,    B  -- 3,    Ca -- 4,    Cu -- 5
# Fe -- 6,    K  -- 7,    Mg -- 8,    Mn -- 9
# Na -- 10,   Ni  -- 11,    P -- 12,  Pb -- 13
# S -- 14,   Zn -- 15
X = data_pp.iloc[:,[0, 1]].values
y = data_pp.iloc[:,[2]].values

# Scaling
X_scaled = stdsc.fit_transform(X)
X_scaled = X

loo = LeaveOneOut()
loo.get_n_splits(X_scaled)
n_samples, n_features = X_scaled.shape

file_object = open(output_filename,'w')   
file_object.write('IterationNumber' + '           MSE' +'              MAE'+'              MAPE'+'              R2'+'              ExpVar'+'              MLSE'+'              MedAE'+'             Model' + '\n')
file_object.close()

# Hyperparameters
alpha = [5, 10]
booster = ['gbtree','gblinear']
colsample_bylevel = [0.1, 1]
colsample_bynode = [0.1, 1]
colsample_bytree = [0.1, 1]
eta = [0.1, 0.3]
gamma = [0, 0.1]
importance_type = ['gain']
learning_rate = [0.3, 0.01]
max_delta_step = [0, 10]
max_depth = [5, 6]
min_child_weight = [1, 4]
n_estimators = [500, 1000]
objective = ['reg:squarederror']
reg_alpha = [0]
reg_lambda = [0.1, 1]
scale_pos_weight = [1, 3]
subsample = [0.5]
random_state = 0
base_score = 0.5
nthread = 4
seed = 0
verbosity = 1 

data1 = []
iteration = 0

for	al	in	alpha:
    for	bt	in	booster:
        for	cl	in	colsample_bylevel:
            for	cn	in	colsample_bynode:
                for	ct	in	colsample_bytree:
                    for	et	in	eta :
                        for	gm	in	gamma:
                            for	it	in	importance_type:
                                for	lr	in	learning_rate:
                                    for	ms	in	max_delta_step:
                                        for	md	in	max_depth:
                                            for	mw	in	min_child_weight:
                                                for	ne	in	n_estimators:
                                                    for	ob	in	objective:
                                                        for	ra	in	reg_alpha:
                                                            for	rl	in	reg_lambda:
                                                                for	sw	in	scale_pos_weight:
                                                                    for	ss	in	subsample:
                                                                        try:
                                                                            
                                                                            predict_loo = []    
                                                                               
                                                                            xgb =  XGBRegressor(alpha=al, 
                                                                                                base_score = base_score, 
                                                                                                booster = bt, 
                                                                                                colsample_bylevel = cl,
                                                                                                colsample_bynode = cn, 
                                                                                                colsample_bytree = ct, 
                                                                                                eta = et,
                                                                                                gamma = gm,
                                                                                                importance_type = it, 
                                                                                                learning_rate = lr, 
                                                                                                max_delta_step = ms,
                                                                                                max_depth = md, 
                                                                                                min_child_weight = mw, 
                                                                                                n_estimators = ne,
                                                                                                nthread = nthread, 
                                                                                                objective = ob, 
                                                                                                random_state = random_state,
                                                                                                reg_alpha = ra, 
                                                                                                reg_lambda = rl, 
                                                                                                scale_pos_weight = sw, 
                                                                                                seed = seed,
                                                                                                subsample = ss,
                                                                                                verbosity = verbosity)
                                                                            
                                                                            for train_index, test_index in loo.split(X_scaled):
                                                                                X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                                                                                y_train, y_test = y[train_index], y[test_index]
                                                                            
                                                                                xgb.fit(X_train, y_train)
                                                                                preds = xgb.predict(X_test)
                                                                                predict_loo.append(round(float(preds), 4))
                                                                                
                                                                            predict_loo_tot = np.array(predict_loo)
                                                                               
                                                                            mse_xgb = np.reshape( mean_squared_error(y, predict_loo_tot), (1,1) )
                                                                            mae_xgb = np.reshape( mean_absolute_error(y, predict_loo_tot), (1,1) )
                                                                            mape_xgb = np.reshape( mean_absolute_percentage_error(y, predict_loo_tot), (1,1) )
                                                                            r2_xgb = np.reshape( r2_score(y, predict_loo_tot), (1,1) )
                                                                            expVar_xgb = np.reshape(explained_variance_score(y, predict_loo_tot), (1,1))
                                                                            msle_xgb = np.reshape(mean_squared_log_error(y, predict_loo_tot), (1,1))
                                                                            medAE_xgb = np.reshape(median_absolute_error(y, predict_loo_tot), (1,1))
                                                                             
                                                                            xgb.fit(X, y)
                                                                            predict_full = np.reshape( xgb.predict(X),(n_samples, 1) )
                                                                            mse_xgb_full = np.reshape( mean_squared_error(y, predict_full), (1,1) )
                                                                            mae_xgb_full = np.reshape( mean_absolute_error(y, predict_full), (1,1) )
                                                                            mape_xgb_full = np.reshape( mean_absolute_percentage_error(y, predict_full), (1,1) )
                                                                            r2_xgb_full = np.reshape( r2_score(y, predict_full), (1,1))
                                                                            expVar_xgb_full = np.reshape(explained_variance_score(y, predict_full), (1,1))
                                                                            msle_xgb_full = np.reshape(mean_squared_log_error(y, predict_full), (1,1))
                                                                            medAE_xgb_full = np.reshape(median_absolute_error(y, predict_full), (1,1))
                                                                        
                                                                            
                                                                            
                                                                            data = {'MSE': mse_xgb, 
                                                                                    'MAE': mae_xgb,
                                                                                    'MAPE': mape_xgb,  
                                                                                    'R2': r2_xgb,
                                                                                    'Explained Var': expVar_xgb,
                                                                                    'MSLE': msle_xgb,
                                                                                    'MedAE': medAE_xgb,
                                                                                    'XGBRegressor': xgb, 
                                                                                    'predicted values': predict_loo_tot}
                                                                            
                                                                            data1.append(data)

                                                                            iteration = iteration + 1
                                                                            print(iteration)

                                                                            if r2_xgb > 0.0:
                                                                                print("XGBregressor LOO:", mse_xgb, mae_xgb, mape_xgb, r2_xgb, expVar_xgb, msle_xgb, medAE_xgb)
                                                                                print("XGBregressor Full:", mse_xgb_full, mae_xgb_full, mape_xgb_full, r2_xgb_full, expVar_xgb_full, msle_xgb_full, medAE_xgb_full)
                                                                                print(xgb)
                                                                                
                                                                                
                                                                            file_object = open(output_filename, 'a')     
                                                                            file_object.write(repr(iteration) + '                   ' + 
                                                                                              repr(round(float(data['MSE']), 5)) + '         ' +
                                                                                              repr(round(float(data['MAE']), 5)) + '         ' +
                                                                                              repr(round(float(data['MAPE']), 5)) + '         ' +
                                                                                              repr(round(float(data['R2']), 5)) + '          ' +
                                                                                              repr(round(float(data['Explained Var']), 5)) + '          ' +
                                                                                             repr(round(float(data['MSLE']), 5)) + '          ' +
                                                                                             repr(round(float(data['MedAE']), 5)) + '          ' +
                                                                                              "".join((str(data['XGBRegressor']).replace("\n","")).split()) + '            '+
                                                                                              str(data['predicted values'].reshape(1,n_samples)).replace("\n"," ")+ '\n' )
                                                                            file_object.close()
                                                                            
                                                                        except:
                                                                            print("Unsuccessful Model: ", xgb)
                                                                            pass

maximum_r2 = []
minimum_mse = []
minimum_mae = []
minimum_mape = []
maximum_expVar = []
minimum_msle = []
minimum_medAE = []

for i in range(len(data1)):
    maximum_r2.append(round(float(data1[i]['R2']), 4))
    minimum_mse.append(round(float(data1[i]['MSE']), 4))
    minimum_mae.append(round(float(data1[i]['MAE']), 4))
    minimum_mape.append(round(float(data1[i]['MAPE']), 4))
    maximum_expVar.append(round(float(data1[i]['Explained Var']), 4))
    minimum_msle.append(round(float(data1[i]['MSLE']), 4))
    minimum_medAE.append(round(float(data1[i]['MedAE']), 4))

print('Largest R2 value:', np.max(maximum_r2))
print('Smallest MSE value:', np.min(minimum_mse))
print('Smallest MAE value:', np.min(minimum_mae))
print('Smallest MAPE value:', np.min(minimum_mape))
print('Largest Explained Var value:', np.max(maximum_expVar))
print('Smallest MSLE value:', np.min(minimum_msle))
print('Smallest MedAE value:', np.min(minimum_medAE))

print('Largest R2 index: ', np.where(maximum_r2 == np.max(maximum_r2)))
print('Smallest MSE index: ', np.where(minimum_mse == np.min(minimum_mse)))
print('Smallest MAE index: ', np.where(minimum_mae == np.min(minimum_mae)))
print('Smallest MAPE index: ', np.where(minimum_mape == np.min(minimum_mape)))
print('Largest Explained Var index: ', np.where(maximum_expVar == np.max(maximum_expVar)))
print('Smallest MSLE index: ', np.where(minimum_msle == np.min(minimum_msle)))
print('Smallest MedAE index: ', np.where(minimum_medAE == np.min(minimum_medAE)))


file_object = open(output_filename, 'a')
file_object.write('R2 : ' + repr(np.max(maximum_r2)) + '\n' +
                  'MSE : ' + repr(np.min(minimum_mse)) + '\n' +
                  'MAE : ' + repr(np.min(minimum_mae)) + '\n' +
                  'MAPE : ' + repr(np.min(minimum_mape)) + '\n' +
                  'Explained Var : ' + repr(np.max(maximum_expVar)) + '\n' +
                  'MSLE : ' + repr(np.min(minimum_msle)) + '\n' +
                  'MedAE : ' + repr(np.min(minimum_medAE)) + '\n' +
                  'R2 indices : ' + repr(np.where(maximum_r2 == np.max(maximum_r2))) + '\n' +
                  'MSE indices : ' + repr(np.where(minimum_mse == np.min(minimum_mse))) + '\n' +
                  'MAE indices : ' + repr(np.where(minimum_mae == np.min(minimum_mae))) + '\n' +
                  'MAPE indices : ' + repr(np.where(minimum_mape == np.min(minimum_mape))) + '\n' +
                  'Explained Var indices : ' + repr(np.where(maximum_expVar == np.max(maximum_expVar))) + '\n'
                  'MSLE indices : ' + repr(np.where(minimum_msle == np.min(minimum_msle))) + '\n'+
                  'MedAE indices : ' + repr(np.where(minimum_medAE == np.min(minimum_medAE)))) 
file_object.close()

print('End of Simulation') 