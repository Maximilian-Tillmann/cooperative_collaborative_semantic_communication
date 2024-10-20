import argparse
import os
import random


import numpy as np
import pickle

import torch

from datasets import load_dataset

from CUSUModel import MultiTaskMultiUserComm, Distributed_MultiTaskMultiUserComm

# Function to run the training and save results
def run_training(iteration, dataset, lr, seed, save_results_to,save_directory,nr_training_epochs, nr_training_epochs_retrain_SU, training_scenario,rotate_images,low_angle,high_angle):
    print(f"Iteration {iteration + 1} - Dataset: {dataset}, Learning Rate: {lr}, Seed: {seed}")

    # Set seed
    random.seed(seed)
    np.random.seed(seed)

    # Get dataset
    first_task_dataset, second_task_dataset, first_test_dataset, second_test_dataset = load_dataset(key=dataset)

    # Train
    save_dir = f"save/{dataset}/"
    save_path = f"save/{dataset}/model_{lr}_{seed}_iter{iteration}"
    os.makedirs(save_dir, exist_ok=True)

    # central case: n_in=784, n_c_latent=64, n_x_latent=16, n_c_h=128, n_x_h=32
    #for distributed case: n_in = 196, n_c_latent = 64/4 = 16, n_x_latent = 4, 
    # n_ohne_CU = 42 --> same number of total parameters
    # n_ohne_CU = n_x_h = 8  --> same specific units
    
    # SU only
    n_ohne_CU = 4
    n_hidden_layer_ohne_CU = 4 # can be set to zero n_ohne_CU = 40 and n_hidden_layer_ohne_CU = 16 gives same number of parametesr as with CU
    

    # CU
    n_c_h = 0 #can be set to zero 32
    n_c_latent = 8 #number of layer for CU output
    
    # SU after CU
    n_x_h = 4 #number of layer for specific unit (input is from CU output)
    n_hidden_layer_su_with_cu = 4 #can be zero, last layer befor output
    

    


    # COnvolutional model number of paramters:
    #\left(\left(\left(9+1\right)k_{1}+\left(\left(9\cdot k_{1}\right)+1\right)k_{2}+2\left(\left(9k_{2}\right)+1\right)k_{3}\right)\right)
    #\frac{x}{x}2\left(\left(9+1\right)s_{1}+\left(\left(9\cdot s_{1}\right)+1\right)s_{2}+\left(\left(9s_{2}\right)+1\right)s_{3}\right)
    

    
    n_x_latent = 2 #number latent variables to be transmitted per distributed agent / per central agent

    # setups with same number of paramters
    broadcast_transmit_CU = False # if true, the output of the common unit is transmitted, if false, the transmissions for the tasks are independent, with just the cu as preprocessing
    centralised_observation = False # if set to true, the whole observation is available at a central node that has to transmit the semantic information
    additional_maxpool = False
    pad = 1

    retrain_SU_for_specific_SNR = True

    # Average Train SNR e.g. 0 or for no noise: +torch.math.inf 
    # k_1 = 4; k_2 = 3; k_3 = 1; s_1 = 3; s_2 = 2; s_3 = 1
    # k_1 =  6; k_2 = 5; k_3 = 2; s_1 = 4; s_2 = 4; s_3 = 2 
    # k_1 = 7; k_2 = 4; k_3 =  4; s_1 = 6; s_2 = 4; s_3 = 2

    # if training_scenario == 'broadcast_vs_no_broadcast_simulation':
    #     SNR_train_range = 4
    #     SNR_task_1 = 7  
    #     SNR_task_2 = 7
    #     if save_results_to == "model_1":
    #         k_1 = 6; k_2 = 5; k_3 = 3; s_1 = 4; s_2 = 4; s_3 = 3      # wenig parameter
    #         centralised_observation = False
    #         broadcast_transmit_CU = False
    #         n_x_latent = 2

    #     elif save_results_to == "model_2":
    #         k_1 =  7; k_2 = 5; k_3 = 4 ; s_1 = 4; s_2 = 4; s_3 = 3    #mehr parameter
    #         centralised_observation = False
    #         broadcast_transmit_CU = True
    #         n_x_latent = 2 

    if training_scenario == 'over_NN_parameters_large':
        print(training_scenario)
        SNR_train_range = 2
        SNR_task_1 = 5  
        SNR_task_2 = 10
        #last Nr. NN was 1750 --> aim for 2300, 2900, 3500
        if save_results_to == "model_1":
            k_1 = 11; k_2 = 11; k_3 = 6; s_1 = 11; s_2 = 7; s_3 = 6     # 2400
            centralised_observation = False
            n_x_latent = 2

        elif save_results_to == "model_2":
            k_1 =  14; k_2 = 12; k_3 = 6; s_1 = 11; s_2 = 9; s_3 = 6    #3000
            centralised_observation = False
            n_x_latent = 2
        
        elif save_results_to == "model_3":
            k_1 = 14; k_2 = 13; k_3 = 8; s_1 = 11; s_2 = 10; s_3 = 8     # 3600
            centralised_observation = False
            n_x_latent = 2
    
        else:
            return

    if training_scenario == 'over_NN_parameters':
        SNR_train_range = 2
        SNR_task_1 = 5  
        SNR_task_2 = 10
        if save_results_to == "model_1":
            k_1 = 4; k_2 = 2; k_3 = 2; s_1 = 2; s_2 = 2; s_3 = 2     # wenig parameter
            centralised_observation = False
            n_x_latent = 2

        elif save_results_to == "model_2":
            k_1 =  4; k_2 = 4; k_3 = 2; s_1 = 3; s_2 = 3; s_3 = 2    #mehr parameter
            centralised_observation = False
            n_x_latent = 2
        
        elif save_results_to == "model_3":
            k_1 = 6; k_2 = 5; k_3 = 3; s_1 = 4; s_2 = 4; s_3 = 3     # mehr parameter 2
            centralised_observation = False
            n_x_latent = 2
    


        elif save_results_to == "model_4":
            k_1 =  8; k_2 = 7; k_3 = 3; s_1 = 8; s_2 = 4; s_3 = 3        #mehr parameter 3
            centralised_observation = False
            n_x_latent = 2


        elif save_results_to == "model_5":
            k_1 = 9; k_2 = 8; k_3 = 4; s_1 = 7; s_2 = 6; s_3 = 4    # mehr parameter 4
            centralised_observation = False
            n_x_latent = 2

        elif save_results_to == "model_6":
            k_1 =  11; k_2 = 9; k_3 = 5; s_1 = 10; s_2 = 6; s_3 = 5    #mehr parameter 5
            centralised_observation = False
            n_x_latent = 2
        

    # Distributed Observation, perfect communication


        elif save_results_to == "model_7":
            k_1 = 4; k_2 = 2; k_3 = 2; s_1 = 2; s_2 = 2; s_3 = 2     # wenig parameter
            centralised_observation = False
            n_x_latent = 2
            SNR_task_1 = np.inf 
            SNR_task_2 = np.inf  

            retrain_SU_for_specific_SNR = False
            nr_training_epochs = nr_training_epochs + nr_training_epochs_retrain_SU

        elif save_results_to == "model_8":
            k_1 =  4; k_2 = 4; k_3 = 2; s_1 = 3; s_2 = 3; s_3 = 2    #mehr parameter
            centralised_observation = False
            n_x_latent = 2
            SNR_task_1 = np.inf 
            SNR_task_2 = np.inf 

            retrain_SU_for_specific_SNR = False
            nr_training_epochs = nr_training_epochs + nr_training_epochs_retrain_SU
        
        elif save_results_to == "model_9":
            k_1 = 6; k_2 = 5; k_3 = 3; s_1 = 4; s_2 = 4; s_3 = 3     # mehr parameter 2
            centralised_observation = False
            n_x_latent = 2
            SNR_task_1 = np.inf 
            SNR_task_2 = np.inf 

            retrain_SU_for_specific_SNR = False
            nr_training_epochs = nr_training_epochs + nr_training_epochs_retrain_SU
    


        elif save_results_to == "model_10":
            k_1 =  8; k_2 = 7; k_3 = 3; s_1 = 8; s_2 = 4; s_3 = 3        #mehr parameter 3
            centralised_observation = False
            n_x_latent = 2
            SNR_task_1 = np.inf 
            SNR_task_2 = np.inf 

            retrain_SU_for_specific_SNR = False
            nr_training_epochs = nr_training_epochs + nr_training_epochs_retrain_SU

        elif save_results_to == "model_11":
            k_1 = 9; k_2 = 8; k_3 = 4; s_1 = 7; s_2 = 6; s_3 = 4    # mehr parameter 4
            centralised_observation = False
            n_x_latent = 2
            SNR_task_1 = np.inf 
            SNR_task_2 = np.inf 

            retrain_SU_for_specific_SNR = False
            nr_training_epochs = nr_training_epochs + nr_training_epochs_retrain_SU

        elif save_results_to == "model_12":
            k_1 =  11; k_2 = 9; k_3 = 5; s_1 = 10; s_2 = 6; s_3 = 5    #mehr parameter 5
            centralised_observation = False
            n_x_latent = 2
            SNR_task_1 = np.inf 
            SNR_task_2 = np.inf 

            retrain_SU_for_specific_SNR = False
            nr_training_epochs = nr_training_epochs + nr_training_epochs_retrain_SU

        else:
            return
        
        # # Central Observation perfect communication
        # elif save_results_to == "model_13":
        #     k_1 = 4; k_2 = 2; k_3 = 2; s_1 = 2; s_2 = 2; s_3 = 2     # wenig parameter
        #     centralised_observation = True
        #     n_x_latent = 20
        #     SNR_task_1 = np.inf 
        #     SNR_task_2 = np.inf 
    
        # elif save_results_to == "model_14":
        #     k_1 =  4; k_2 = 4; k_3 = 2; s_1 = 3; s_2 = 3; s_3 = 2    #mehr parameter
        #     centralised_observation = True
        #     n_x_latent = 20
        #     SNR_task_1 = np.inf 
        #     SNR_task_2 = np.inf 

        # if save_results_to == "model_15":
        #     k_1 = 6; k_2 = 5; k_3 = 3; s_1 = 4; s_2 = 4; s_3 = 3     # mehr parameter 2
        #     centralised_observation = True
        #     n_x_latent = 20
        #     SNR_task_1 = np.inf 
        #     SNR_task_2 = np.inf 


        # elif save_results_to == "model_16":
        #     k_1 =  8; k_2 = 7; k_3 = 3; s_1 = 8; s_2 = 4; s_3 = 3        #mehr parameter 3
        #     centralised_observation = True
        #     n_x_latent = 20
        #     SNR_task_1 = np.inf 
        #     SNR_task_2 = np.inf 

        # elif save_results_to == "model_17":
        #     k_1 = 9; k_2 = 8; k_3 = 4; s_1 = 7; s_2 = 6; s_3 = 4    # mehr parameter 4
        #     centralised_observation = True
        #     n_x_latent = 20
        #     SNR_task_1 = np.inf 
        #     SNR_task_2 = np.inf 

        # elif save_results_to == "model_18":
        #     k_1 =  11; k_2 = 9; k_3 = 5; s_1 = 10; s_2 = 6; s_3 = 5    #mehr parameter 5
        #     centralised_observation = True
        #     n_x_latent = 20
        #     SNR_task_1 = np.inf 
        #     SNR_task_2 = np.inf 





    elif training_scenario == 'over_SNR':
        SNR_train_range = 2

        k_1 = 6; k_2 = 5; k_3 = 3; s_1 = 4; s_2 = 4; s_3 = 3     # mehr parameter 2
        centralised_observation = False
        n_x_latent = 2

        if save_results_to == 'model_1':
            SNR_task_1 = -11 
            SNR_task_2 = -11
        
        elif save_results_to == 'model_2':
            SNR_task_1 = -8  
            SNR_task_2 = -8

        elif save_results_to == 'model_3':  #--> use this scenatio for CUSU vs CUSUwithRetrain and show the Training iteration curve here!
            SNR_task_1 = -5  
            SNR_task_2 = -5

        elif save_results_to == 'model_4':
            SNR_task_1 = -2 
            SNR_task_2 = -2

        elif save_results_to == 'model_5':
            SNR_task_1 = 1  
            SNR_task_2 = 1

        elif save_results_to == 'model_6':
            SNR_task_1 = 4  
            SNR_task_2 = 4
        
        elif save_results_to == 'model_7':
            SNR_task_1 = 7  
            SNR_task_2 = 7

        elif save_results_to == 'model_8':
            SNR_task_1 = 10  
            SNR_task_2 = 10

        elif save_results_to == 'model_9':
            SNR_task_1 = 13  
            SNR_task_2 = 13

        elif save_results_to == 'model_10':
            SNR_task_1 = 16  
            SNR_task_2 = 16

        elif save_results_to == 'model_11':
            SNR_task_1 = 19  
            SNR_task_2 = 19
        
        
        elif save_results_to == 'model_12':
            SNR_task_1 = 5 #4.5  
            SNR_task_2 = 5 #4.5

            SNR_train_range = 30# 19
            retrain_SU_for_specific_SNR = False
            nr_training_epochs = nr_training_epochs + nr_training_epochs_retrain_SU

        elif save_results_to == 'model_13':  
            SNR_task_1 = 10 
            SNR_task_2 = 10

            retrain_SU_for_specific_SNR = False
            nr_training_epochs = nr_training_epochs + nr_training_epochs_retrain_SU
        else:
            return
    

    elif training_scenario == 'pretrain_CU_for_perfect_ch':
        SNR_train_range = 2

        k_1 = 6; k_2 = 5; k_3 = 3; s_1 = 4; s_2 = 4; s_3 = 3     # mehr parameter 2
        centralised_observation = False
        n_x_latent = 2

        if save_results_to == 'model_1':
            SNR_task_1 = 10 
            SNR_task_2 = 10
        else:
            return

    model = Distributed_MultiTaskMultiUserComm(n_in=196, n_c_latent=n_c_latent, n_x_latent=n_x_latent, n_c_h=n_c_h, n_x_h=n_x_h, n_ohne_CU=n_ohne_CU, n_hidden_layer_ohne_CU=n_hidden_layer_ohne_CU,n_hidden_layer_su_with_cu=n_hidden_layer_su_with_cu, k_1=k_1, k_2=k_2, k_3=k_3, s_1=s_1, s_2=s_2, s_3=s_3,transmit_CU=broadcast_transmit_CU, centralised_observation=centralised_observation,SNR_task1=SNR_task_1,SNR_task2=SNR_task_2,additional_maxpool=additional_maxpool,pad=pad,SNR_train_range=SNR_train_range)


    if retrain_SU_for_specific_SNR == True:
        # set snr to train model for all SNR: 
        model.multitask_network.SNR_task_1 = 4.5 #np.inf 
        model.multitask_network.SNR_task_2 = 4.5 #np.inf 
        model.multitask_network.SNR_train_range = 19

        if training_scenario == 'pretrain_CU_for_perfect_ch':
            model.multitask_network.SNR_task_1 = np.inf 
            model.multitask_network.SNR_task_2 = np.inf 
            model.multitask_network.SNR_train_range = 0


    print(f"Model: {type(model)}")

    model.fit(first_dataset=first_task_dataset, second_dataset=second_task_dataset, first_test_dataset=first_test_dataset,
              second_test_dataset=second_test_dataset, batch_size=100, n_epoch_primal=nr_training_epochs, learning_rate=lr, path=save_path, iteration=iteration,rotate_images=rotate_images,low_angle=low_angle,high_angle=high_angle)  #n_epoch_primal=400, 500

    if retrain_SU_for_specific_SNR == False:
        model.fit(first_dataset=first_task_dataset, second_dataset=second_task_dataset, first_test_dataset=first_test_dataset,
              second_test_dataset=second_test_dataset, batch_size=100, n_epoch_primal=10, learning_rate=lr*0.1, path=save_path, iteration=iteration,rotate_images=rotate_images,low_angle=low_angle,high_angle=high_angle)
        model.fit(first_dataset=first_task_dataset, second_dataset=second_task_dataset, first_test_dataset=first_test_dataset,
              second_test_dataset=second_test_dataset, batch_size=100, n_epoch_primal=10, learning_rate=lr*0.01, path=save_path, iteration=iteration,rotate_images=rotate_images,low_angle=low_angle,high_angle=high_angle)
        model.fit(first_dataset=first_task_dataset, second_dataset=second_task_dataset, first_test_dataset=first_test_dataset,
              second_test_dataset=second_test_dataset, batch_size=100, n_epoch_primal=10, learning_rate=lr*0.001, path=save_path, iteration=iteration,rotate_images=rotate_images,low_angle=low_angle,high_angle=high_angle)

    # obj0, obj1, obj2 are created here...


    
    


    # NOW Train SU for different channels, while not retraining the CU:
    if retrain_SU_for_specific_SNR == True:
        

        #freeze weights of CU:
        model.multitask_network.cu_agent_1.weight.required_grad = False
        model.multitask_network.conv_2_agent_1.weight.required_grad = False

        model.multitask_network.cu_agent_2.weight.required_grad = False
        model.multitask_network.conv_2_agent_2.weight.required_grad = False

        model.multitask_network.cu_agent_3.weight.required_grad = False
        model.multitask_network.conv_2_agent_3.weight.required_grad = False

        model.multitask_network.cu_agent_4.weight.required_grad = False
        model.multitask_network.conv_2_agent_4.weight.required_grad = False

        # Set new training SNR: 
        model.multitask_network.SNR_task_1 = SNR_task_1
        model.multitask_network.SNR_task_2 = SNR_task_2


        model.fit(first_dataset=first_task_dataset, second_dataset=second_task_dataset, first_test_dataset=first_test_dataset,
              second_test_dataset=second_test_dataset, batch_size=100, n_epoch_primal=nr_training_epochs_retrain_SU, learning_rate=lr, path=save_path, iteration=iteration,rotate_images=rotate_images,low_angle=low_angle,high_angle=high_angle)  #n_epoch_primal=400, 500
        model.fit(first_dataset=first_task_dataset, second_dataset=second_task_dataset, first_test_dataset=first_test_dataset,
              second_test_dataset=second_test_dataset, batch_size=100, n_epoch_primal=10, learning_rate=lr*0.1, path=save_path, iteration=iteration,rotate_images=rotate_images,low_angle=low_angle,high_angle=high_angle)  #n_epoch_primal=400, 500
        model.fit(first_dataset=first_task_dataset, second_dataset=second_task_dataset, first_test_dataset=first_test_dataset,
              second_test_dataset=second_test_dataset, batch_size=100, n_epoch_primal=10, learning_rate=lr*0.01, path=save_path, iteration=iteration,rotate_images=rotate_images,low_angle=low_angle,high_angle=high_angle)
        model.fit(first_dataset=first_task_dataset, second_dataset=second_task_dataset, first_test_dataset=first_test_dataset,
              second_test_dataset=second_test_dataset, batch_size=100, n_epoch_primal=10, learning_rate=lr*0.001, path=save_path, iteration=iteration,rotate_images=rotate_images,low_angle=low_angle,high_angle=high_angle)
        
        
    # Save Results
    accuracy_task_1 = np.array(torch.tensor(model.accuracy_task_1, device = 'cpu')) 
    accuracy_task_2 = np.array(torch.tensor(model.accuracy_task_2, device = 'cpu')) 
    accuracy_task_1_ohnecu = np.array(torch.tensor(model.accuracy_task_1_ohnecu, device = 'cpu')) 
    accuracy_task_2_ohnecu = np.array(torch.tensor(model.accuracy_task_2_ohnecu, device = 'cpu')) 

    train_loss_su1 = np.array(torch.tensor(model.train_loss_su1, device = 'cpu')) 
    train_loss_su2 = np.array(torch.tensor(model.train_loss_su2, device = 'cpu')) 
    train_loss_su_ohnecu = np.array(torch.tensor(model.train_loss_su_ohnecu, device = 'cpu')) 
    train_loss_su_ohnecu_task2 = np.array(torch.tensor(model.train_loss_su_ohnecu_task2, device = 'cpu')) 

    error_task_1 = np.array(torch.tensor(model.error_task_1, device = 'cpu')) 
    error_task_2 = np.array(torch.tensor(model.error_task_2, device = 'cpu')) 
    error_task_1_ohnecu = np.array(torch.tensor(model.error_task_1_ohnecu, device = 'cpu')) 
    error_task_2_ohnecu = np.array(torch.tensor(model.error_task_2_ohnecu, device = 'cpu')) 

    error_task_1_mitcu_overSNR = np.array(torch.tensor(model.error_task_1_mitcu_overSNR, device = 'cpu')) 
    error_task_2_mitcu_overSNR = np.array(torch.tensor(model.error_task_2_mitcu_overSNR, device = 'cpu')) 
    error_task_1_ohnecu_overSNR = np.array(torch.tensor(model.error_task_1_ohnecu_overSNR, device = 'cpu')) 
    error_task_2_ohnecu_overSNR = np.array(torch.tensor(model.error_task_2_ohnecu_overSNR, device = 'cpu')) 


    # Save numpy files
    os.makedirs("dual-cooperative-semantic-main/results/"+save_directory+"/"+save_results_to+ "/", exist_ok=True)

    # Saving the objects:
    with open('dual-cooperative-semantic-main/results/'+save_directory+"/"+save_results_to+'/objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([k_1, k_2, k_3, s_1, s_2, s_3, n_x_latent, broadcast_transmit_CU, centralised_observation, SNR_task_1,SNR_task_2],f)

    np.save(f"dual-cooperative-semantic-main/results/"+save_directory+"/"+save_results_to+ "/exp_{dataset}_train_loss_{lr}_{seed}_iter{iteration}.npy", np.array(model.train_losses))
    np.save(f"dual-cooperative-semantic-main/results/"+save_directory+"/"+save_results_to+ "/exp_{dataset}_train_time_{lr}_{seed}_iter{iteration}.npy", np.array(model.train_times))

    # Saving Accuracy outputs for later Modifications
    accuracy_save_dir = 'dual-cooperative-semantic-main/results/'+save_directory+"/"+save_results_to+ '/accuracy_outputs'
    os.makedirs(accuracy_save_dir, exist_ok=True)
    np.save(os.path.join(accuracy_save_dir, f'accuracy_task_1_iter{iteration}.npy'), accuracy_task_1)
    np.save(os.path.join(accuracy_save_dir, f'accuracy_task_2_iter{iteration}.npy'), accuracy_task_2)
    np.save(os.path.join(accuracy_save_dir, f'accuracy_task_1_ohnecu_iter{iteration}.npy'), accuracy_task_1_ohnecu)
    np.save(os.path.join(accuracy_save_dir, f'accuracy_task_2_ohnecu_iter{iteration}.npy'), accuracy_task_2_ohnecu)

    # Saving loss values and time
    training_save_dir = 'dual-cooperative-semantic-main/results/'+save_directory+"/"+save_results_to+ '/training_losses'
    os.makedirs(training_save_dir, exist_ok=True)
    np.save(os.path.join(training_save_dir, f'train_loss_su1_iter{iteration}.npy'), train_loss_su1)
    np.save(os.path.join(training_save_dir, f'train_loss_su2_iter{iteration}.npy'), train_loss_su2)
    np.save(os.path.join(training_save_dir, f'train_loss_su_ohnecu_iter{iteration}.npy'), train_loss_su_ohnecu)
    np.save(os.path.join(training_save_dir, f'train_loss_su_ohnecu_iter_task2{iteration}.npy'), train_loss_su_ohnecu_task2)

    # Saving error rates (These are not in percentage)
    errorrate_save_dir = 'dual-cooperative-semantic-main/results/'+save_directory+"/"+save_results_to+ '/error_rates'
    os.makedirs(errorrate_save_dir, exist_ok=True)
    np.save(os.path.join(errorrate_save_dir, f'error_task_1_iter{iteration}.npy'), error_task_1)
    np.save(os.path.join(errorrate_save_dir, f'error_task_2_iter{iteration}.npy'), error_task_2)
    np.save(os.path.join(errorrate_save_dir, f'error_task_1_ohnecu_iter{iteration}.npy'), error_task_1_ohnecu)
    np.save(os.path.join(errorrate_save_dir, f'error_task_2_ohnecu_iter{iteration}.npy'), error_task_2_ohnecu)

    # save error tates over SNR after training finished
    SNR_save_dir = 'dual-cooperative-semantic-main/results/'+save_directory+"/"+save_results_to+ '/error_rates_over_SNR'
    os.makedirs(SNR_save_dir, exist_ok=True)
    np.save(os.path.join(SNR_save_dir, f'error_task_1_iter{iteration}.npy'), error_task_1_mitcu_overSNR)
    np.save(os.path.join(SNR_save_dir, f'error_task_2_iter{iteration}.npy'), error_task_2_mitcu_overSNR)
    np.save(os.path.join(SNR_save_dir, f'error_task_1_ohnecu_iter{iteration}.npy'), error_task_1_ohnecu_overSNR)
    np.save(os.path.join(SNR_save_dir, f'error_task_2_ohnecu_iter{iteration}.npy'), error_task_2_ohnecu_overSNR)

if __name__ == '__main__':
    
    save_directory = "run_37" 

    # run_32 main simulation without image roation
    # run_34 with image rotation 30 degree
    # run_32 for train CU perfect channel, then retrain SU for channel coding (show this does not work well)
    # run_35_NN :  no image rotation, over NN task1: 5dB, task2: 10dB
    # run_36_SNR :  with image rotation 20 degree
    # run_37_SNR / 37_NN :  with image rotation 30 degree
    training_scenario = 'over_NN_parameters_large'  # 'over_NN_parameters' , 'over_SNR', pretrain_CU_for_perfect_ch

    save_results_to = "model_1" # if train_slurm = True does not matter, which is select here

    start_iteration_to_save = 20 #first new data is saved as "start_iteration_to_save" 

    train_slurm = False  # True False
    

    nr_training_epochs = 250   #250,   450 is enough
    nr_training_epochs_retrain_SU = 220# 220

    nr_training_iterations = 5 #(at least 10 for final run)  #in total 600 iteration took about 6 hours, meaning that 100 iterations = 1hour

    

    rotate_images=True # True or False
    low_angle=-30  # was +-30 degree for run 34, for run 36 its +-15 degree
    high_angle=+30



    if training_scenario == 'over_NN_parameters':
        save_directory = save_directory + "_NN"
    elif training_scenario == 'over_NN_parameters_large':
        save_directory = save_directory + "_NN_large"
    elif training_scenario == 'over_SNR':
        save_directory = save_directory + "_SNR"
    elif training_scenario == 'pretrain_CU_for_perfect_ch':
        save_directory = save_directory + "_pretrain_CU_for_perfect_ch"

    parser = argparse.ArgumentParser(description="Semantic Common Unit with Implicit Optimal Priors.")

    if train_slurm == True:
        parser.add_argument('--iteration_id', dest='iteration_id', type=str, help='Add id of current simulation instance')
        args = parser.parse_args()
        # print(args.iteration_id)
        # print('datatype='+str(type(args.iteration_id)))

        save_results_to = "model_" + args.iteration_id
        print("\n save_directory:"+ save_directory +"\n")
        print("run model:" + save_results_to )

    
    
    # parser.add_argument("--dataset", type=str, default="MNIST", help="Dataset Name.")
    #parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning Rate for CU.")
    # parser.add_argument("--seed", type=int, default=42, help="Random Seed.")

    

    # dataset = args.dataset
    # lr = args.learning_rate
    # seed = args.seed
    lr = 1e-4
    seed = 43 #42
    dataset = "MNIST"

    
    # Run training for 20 iterations
    for i in range(nr_training_iterations): #at least 5, better 25 , for "run_17 was 50 iterations, and n_epoch_primal=450"
        run_training(i+start_iteration_to_save, dataset, lr, seed, save_results_to,save_directory,nr_training_epochs, nr_training_epochs_retrain_SU,training_scenario,rotate_images,low_angle,high_angle)

    print("training finished")