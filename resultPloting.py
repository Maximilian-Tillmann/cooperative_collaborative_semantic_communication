import os
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import pickle


# Function to load .npy files and calculate average
def load_and_average(directory, file_prefix):
    file_list = [f for f in os.listdir(directory) if f.startswith(file_prefix) and f.endswith('.npy')]
    data = []
    n_counter = 0
    for file_name in file_list:
        n_counter = n_counter + 1
        #if n_counter < 11:
        data.append(np.load(os.path.join(directory, file_name)))
    
    #filtered_list = [row for row in data if row[-1]/row[0] < 0.99] #exclude outliers where training did not converge at all
    # filtered_list = [row for row in data if row[-1] < np.mean(np.transpose(data)[-1])]
    # del data[16]
    filtered_list = data
    # filtered_list = data
    return np.mean(np.array(filtered_list), axis=0)


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


# run_14_SNR_train_range_4 vs run_11 (5 iterations) 
# run_17_SNR_task1is0_SNR_task2is10 or run_17_SNR_task1is10_SNR_task2is10 

# run_22_broadcast_Same_SNR or run_21_broadcast

load_directory = "run_37_NN" #run_34_SNR and run_32_SNR, run_32_NN # run_37 : SNR, NN , NN_large 
load_model = "model_1"  #1,  7,  13 

plot_error_over_epochs = True
plot_over_SNR = True
SNR = '0'   # 'inf' or '0'

if load_directory[-2:] == 'NN' or load_directory[-2:] == 'ge':
    Plot_over_Nr_parameters = True  #True #False
    plot_task_1_task_2 = True
    plot_SNR_retrain = False
else:
    Plot_over_Nr_parameters = False  #True #False
    plot_task_1_task_2 = False
    plot_SNR_retrain = True



# Getting back the objects:
with open('dual-cooperative-semantic-main/results/'+load_directory+"/"+load_model+'/objs.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    k_1, k_2, k_3, s_1, s_2, s_3, n_x_latent, transmit_CU, centralised_observation, SNR_task_1,SNR_task_2 = pickle.load(f)

accuracy_dir = 'dual-cooperative-semantic-main/results/'+load_directory+"/"+load_model+'/accuracy_outputs'
error_dir = 'dual-cooperative-semantic-main/results/'+load_directory+"/"+load_model+'/error_rates'
training_dir = 'dual-cooperative-semantic-main/results/'+load_directory+"/"+load_model+'/training_losses'
SNR_save_dir = 'dual-cooperative-semantic-main/results/'+load_directory+"/"+load_model+ '/error_rates_over_SNR'

os.makedirs('dual-cooperative-semantic-main/plots_CU_tikz/'+load_directory+"/"+load_model, exist_ok=True)
os.makedirs('dual-cooperative-semantic-main/plots_CU_png/'+load_directory+"/"+load_model, exist_ok=True)

# Load and average accuracy_task_1 from all iterations
avg_accuracy_task_1 = load_and_average(accuracy_dir, 'accuracy_task_1_iter')
avg_accuracy_task_1_ohnecu = load_and_average(accuracy_dir, 'accuracy_task_1_ohnecu_iter')
avg_accuracy_task_2 = load_and_average(accuracy_dir, 'accuracy_task_2_iter')
avg_accuracy_task_2_ohnecu = load_and_average(accuracy_dir, 'accuracy_task_2_ohnecu_iter')

avg_error_task_1 = load_and_average(error_dir,'error_task_1_iter')
avg_error_task_1_ohnecu = load_and_average(error_dir,'error_task_1_ohnecu_iter')
avg_error_task_2 = load_and_average(error_dir, 'error_task_2_iter')
avg_error_task_2_ohnecu = load_and_average(error_dir, 'error_task_2_ohnecu_iter')

avg_train_loss_su1 = load_and_average(training_dir, 'train_loss_su1_iter')
avg_train_loss_su_ohnecu = load_and_average(training_dir, 'train_loss_su_ohnecu_iter')





            
avg_error_task_1_mitcu_overSNR = load_and_average(SNR_save_dir, 'error_task_1_iter')
avg_error_task_2_mitcu_overSNR = load_and_average(SNR_save_dir, 'error_task_2_iter')
avg_error_task_1_ohnecu_overSNR = load_and_average(SNR_save_dir, 'error_task_1_ohnecu_iter')
avg_error_task_2_ohnecu_overSNR = load_and_average(SNR_save_dir, 'error_task_2_ohnecu_iter')

save_formats = ['pdf', 'png', 'eps']



#  colors
indigo = tuple(np.array([51, 34, 136]) / 255)
cyan_muted = tuple(np.array([136, 204, 238]) / 255)
teal_muted = tuple(np.array([68, 170, 153]) / 255)
green = tuple(np.array([17, 119, 51]) / 255)
olive = tuple(np.array([153, 153, 51]) / 255)
sand = tuple(np.array([221, 204, 119]) / 255)
rose = tuple(np.array([204, 102, 119]) / 255)
wine = tuple(np.array([136, 34, 85]) / 255)
purple = tuple(np.array([170, 68, 153]) / 255)

orange = tuple(np.array([238, 119, 51]) / 255)
red = tuple(np.array([204, 51, 17]) / 255)
blue = tuple(np.array([0, 119, 187]) / 255)

# rottöne für ohne CU
# Task1: orange, rose, Task2: red, wine,               

#blau/grün für mit CU
# Task1: blue, indigo, cyan_muted, purple      Task2: teal_muted, green, olive, sand


label_task1_CUSU_retrainSmallSNRrange = 'task1: CU and SU, SU retrained for small SNR range'
color_task1_CUSU_retrainSmallSNRrange = indigo #blue
marker_task1_CUSU_retrainSmallSNRrange = 'o'

label_task2_CUSU_retrainSmallSNRrange = 'task2: CU and SU, SU retrained for small SNR range'
color_task2_CUSU_retrainSmallSNRrange = indigo #teal_muted
marker_task2_CUSU_retrainSmallSNRrange = 'o' #'s'



label_task1_ohne_CU_retrainSmallSNRrange = 'task1: without CU, SU retrained for small SNR range'
color_task1_ohne_CU_retrainSmallSNRrange = wine#orange
marker_task1_ohne_CU_retrainSmallSNRrange = 'd'

label_task2_ohne_CU_retrainSmallSNRrange = 'task2: without CU, SU retrained for small SNR range'
color_task2_ohne_CU_retrainSmallSNRrange = wine #red
marker_task2_ohne_CU_retrainSmallSNRrange = 'd' #'>'



label_task1_CUSU_trainFullSNRrange = 'task1: CU and SU, trained for full SNR range'
color_task1_CUSU_trainFullSNRrange = teal_muted #indigo
marker_task1_CUSU_trainFullSNRrange =  's'  # '<'

label_task2_CUSU_trainFullSNRrange = 'task2: CU and SU, trained for full SNR range'
color_task2_CUSU_trainFullSNRrange = teal_muted #green
marker_task2_CUSU_trainFullSNRrange = 's'  #'*'


label_task1_CUSU_pretrain_perfect_ch = "task1: pretrain CU for perfect channel"
color_task1_CUSU_pretrain_perfect_ch = green

label_task2_CUSU_pretrain_perfect_ch = "task2: pretrain CU for perfect channel"
color_task2_CUSU_pretrain_perfect_ch = green




label_task1_ohne_CU_trainFullSNRrange = 'task1: without CU, SU trained for full SNR range'
color_task1_ohne_CU_trainFullSNRrange = purple# rose
marker_task1_ohne_CU_trainFullSNRrange = '^'#'1'

label_task2_ohne_CU_trainFullSNRrange = 'task2: without CU, SU trained for full SNR range'
color_task2_ohne_CU_trainFullSNRrange = purple# wine
marker_task2_ohne_CU_trainFullSNRrange = '^'#'p'


label_task1_CUSU_TrainSmallSNRrange = 'task1: CU and SU, Trained for small SNR range'
color_task1_CUSU_TrainSmallSNRrange = sand
marker_task1_CUSU_TrainSmallSNRrange = 'x'

label_task2_CUSU_TrainSmallSNRrange = 'task2: CU and SU, Trained for small SNR range'
color_task2_CUSU_TrainSmallSNRrange = sand
marker_task2_CUSU_TrainSmallSNRrange = '^'



label_task1_CUSU_noChNoise = 'task1: CU and SU, no channel noise'
color_task1_CUSU_noChNoise = cyan_muted
marker_task1_CUSU_noChNoise = '<'

label_task2_CUSU_noChNoise = 'task2: CU and SU, no channel noise'
color_task2_CUSU_noChNoise = cyan_muted
marker_task2_CUSU_noChNoise = '>'



label_task1_ohne_CU_noChNoise = 'task1: without CU, no channel noise'
color_task1_ohne_CU_noChNoise = rose
marker_task1_ohne_CU_noChNoise = '^'

label_task2_ohne_CU_noChNoise = 'task2: without CU, no channel noise'
color_task2_ohne_CU_noChNoise = rose
marker_task2_ohne_CU_noChNoise = 'x'






#plt.figure(figsize=(10, 5))
#plt.figure()
#plt.plot(self.train_losses, label='System Training Loss')
#plt.title('System Training Loss Convergence')
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.legend()
#plt.grid(True)
#tikzplotlib.save("Total_loss_plot.tikz")
#for format in save_formats:
#    filename = f'system_training_loss_plot.{format}'
#    plt.savefig(filename, format=format, bbox_inches='tight', dpi=300)
#plt.show()

#combined plot of training losses of specific units
# plt.figure()
# plt.semilogy(avg_train_loss_su1, label='SU Training Loss (With CU)', color= 'blue', marker='o')
# plt.semilogy(avg_train_loss_su_ohnecu, label='SU Training Loss (Without CU)', color= 'red', marker='x')
# #plt.title('Training Loss Convergence')
# plt.xlabel('Epoch')
# plt.ylabel('Losses')
# plt.legend()
# plt.grid(True)
# #tikzplotlib.save("Training_loss_plot.tikz")
# # for format in save_formats:
# #     filename = f'combined_training_loss_plot.{format}'
# #     plt.savefig(filename, format=format, bbox_inches='tight', dpi=300)
# plt.show()

# Ploting logarithmic error rates
if plot_error_over_epochs == True:
    fig = plt.figure()
    plt.semilogy(avg_error_task_1, label='Task1 error rate (With CU)', color=cyan_muted, marker='o',markerfacecolor="None")
    plt.semilogy(avg_error_task_2, label='Task2 error rate (With CU)', color=teal_muted, marker='s',markerfacecolor="None")
    plt.semilogy(avg_error_task_1_ohnecu, label='Task1 error rate (Without CU)', color=wine, marker='^',markerfacecolor="None")
    plt.semilogy(avg_error_task_2_ohnecu, label='Task2 error rate (Without CU)', color=purple, marker='d',markerfacecolor="None")
    # plt.title('Impact of the CU')
    plt.xlabel('Epoch')
    plt.ylabel('Tasks execution error rate')
    plt.legend()
    plt.grid(True)


    # save plots

    tikzplotlib_fix_ncols(fig) #needed because of bug caused by updated matplotlib that renamed ncols
    tikzplotlib.save('dual-cooperative-semantic-main/plots_CU_tikz/'+load_directory+"/"+load_model+'/error.tikz', override_externals=True, encoding='utf-8', strict=True)



    plt.title("k_1="+str(k_1)+", k_2="+str(k_2)+", k_3="+str(k_3)+", s_1="+str(s_1)+", s_2="+str(s_2)+", s_3="+str(s_3)+ ", SNR task1="+str(SNR_task_1)+ ", SNR task2="+str(SNR_task_2)+", Ch.uses="+str(n_x_latent)+"\n"+"combined transmission from CU:" +str(transmit_CU)+", central observer:"+str(centralised_observation))
    fig.savefig('dual-cooperative-semantic-main/plots_CU_png/'+load_directory+"/"+load_model+'/error.png', dpi=fig.dpi)

    plt.show()
        



# Plot of accuracies
if plot_error_over_epochs == True:
    fig = plt.figure()
    plt.plot(avg_accuracy_task_1, label='Accuracy of Task1 (With CU)', color=cyan_muted, marker='o',markerfacecolor="None")
    plt.plot(avg_accuracy_task_2, label='Accuracy of Task2 (With CU)', color=teal_muted, marker='s',markerfacecolor="None")
    plt.plot(avg_accuracy_task_1_ohnecu, label='Accuracy of Task1 (Without CU)', color=wine, marker='^',markerfacecolor="None")
    plt.plot(avg_accuracy_task_2_ohnecu, label='Accuracy of Task2 (Without CU)', color=purple, marker='d',markerfacecolor="None")
    #plt.title('Accuracy of Task Execution (Impact of CU)')
    plt.xlabel('Epoch')
    plt.ylabel('Tasks execution accuracy (%)')
    plt.legend()
    plt.grid(True)

    # save plots

    tikzplotlib_fix_ncols(fig) #needed because of bug caused by updated matplotlib that renamed ncols
    tikzplotlib.save('dual-cooperative-semantic-main/plots_CU_tikz/'+load_directory+"/"+load_model+'/accuracy.tikz', override_externals=True, encoding='utf-8', strict=True)



    plt.title("k_1="+str(k_1)+", k_2="+str(k_2)+", k_3="+str(k_3)+", s_1="+str(s_1)+", s_2="+str(s_2)+", s_3="+str(s_3)+", SNR task1="+str(SNR_task_1)+ ", SNR task2="+str(SNR_task_2)+", Ch.uses="+str(n_x_latent)+"\n"+"combined transmission from CU:" +str(transmit_CU)+", central observer:"+str(centralised_observation))
    fig.savefig('dual-cooperative-semantic-main/plots_CU_png/'+load_directory+"/"+load_model+'/accuracy.png', dpi=fig.dpi)

    plt.show()

# Plot of accuricies logarithmic scale
#plt.figure()
#plt.semilogy(avg_accuracy_task_1, label='Accuracy of Task1 (With CU)', color='blue', alpha=0.2, marker='o')
#plt.semilogy(avg_accuracy_task_1_ohnecu, label='Accuracy of Task1 (Without CU)', color='red', alpha=0.2, marker='x')
#plt.title('Accuracy of Task Execution (Impact of CU)')
#plt.xlabel('Epoch')
#plt.ylabel('Tasks execution accuracy')
#plt.legend()
#tikzplotlib.save("log_Tasks_accuracies_plot.tikz")
#for format in save_formats:
#    filename = f'log_tasks_accuracies_time.{format}'
#    plt.savefig(filename, format=format, bbox_inches='tight', dpi=300)
#plt.show()




#Plot result over SNR:

Nr_SNR_evaluations = 41
SNR_eval_min = -20
SNR_eval_max = +20
SNR_eval_linspace = np.linspace(SNR_eval_min,SNR_eval_max,Nr_SNR_evaluations)
# Plot of accuracies
if plot_over_SNR == True:
    fig = plt.figure()
    plt.semilogy(SNR_eval_linspace,avg_error_task_1_mitcu_overSNR[-Nr_SNR_evaluations:], label='Error of Task1 (With CU)', color=cyan_muted, marker='o',markerfacecolor="None")
    plt.semilogy(SNR_eval_linspace,avg_error_task_2_mitcu_overSNR[-Nr_SNR_evaluations:], label='Error of Task2 (With CU)', color=teal_muted, marker='s',markerfacecolor="None")
    plt.semilogy(SNR_eval_linspace,avg_error_task_1_ohnecu_overSNR[-Nr_SNR_evaluations:], label='Error of Task1 (Without CU)', color=wine, marker='^',markerfacecolor="None")
    plt.semilogy(SNR_eval_linspace,avg_error_task_2_ohnecu_overSNR[-Nr_SNR_evaluations:], label='Error of Task2 (Without CU)', color=purple, marker='d',markerfacecolor="None")
    plt.semilogy(SNR_eval_linspace,avg_error_task_1_mitcu_overSNR[-Nr_SNR_evaluations:]/avg_error_task_1_ohnecu_overSNR[-Nr_SNR_evaluations:], label='Task1 difference', color=sand, marker='<',markerfacecolor="None")
    plt.semilogy(SNR_eval_linspace,avg_error_task_2_mitcu_overSNR[-Nr_SNR_evaluations:]/avg_error_task_2_ohnecu_overSNR[-Nr_SNR_evaluations:], label='Task2 difference', color=green, marker='>',markerfacecolor="None")
    #plt.title('Accuracy of Task Execution (Impact of CU)')
    plt.xlabel('SNR')
    plt.ylabel('Error rate')
    plt.legend()
    plt.ylim(0.005, 1)
    plt.grid(True)

    # save plots

    tikzplotlib_fix_ncols(fig) #needed because of bug caused by updated matplotlib that renamed ncols
    tikzplotlib.save('dual-cooperative-semantic-main/plots_CU_tikz/'+load_directory+"/"+load_model+'/snr.tikz', override_externals=True, encoding='utf-8', strict=True)


    plt.title("k_1="+str(k_1)+", k_2="+str(k_2)+", k_3="+str(k_3)+", s_1="+str(s_1)+", s_2="+str(s_2)+", s_3="+str(s_3)+ ", SNR task1="+str(SNR_task_1)+", SNR task2="+str(SNR_task_2)+", Ch.uses="+str(n_x_latent)+"\n"+"combined transmission from CU:" +str(transmit_CU)+", central observer:"+str(centralised_observation))
    fig.savefig('dual-cooperative-semantic-main/plots_CU_png/'+load_directory+"/"+load_model+'/snr.png', dpi=fig.dpi)

    plt.show()


if Plot_over_Nr_parameters == True:

    Nr_parameter_plot = 6
    if load_directory == "run_37_NN_large":
        Nr_parameter_plot = 3
    

    eval_at_SNR_index = 25 # 25 gives 5 SNR for task 1
    eval_at_SNR_index_task_2 = 30 #30 gives 10 SNR  for task 2

    SNR_eval = SNR_eval_linspace[eval_at_SNR_index]

    Nr_of_parameters_withCU = np.zeros((Nr_parameter_plot))
    Nr_of_parameters_ohneCU = np.zeros((Nr_parameter_plot))

    error_task_1_mitcu_over_NrParam = np.ones((Nr_parameter_plot))
    error_task_2_mitcu_over_NrParam = np.ones((Nr_parameter_plot))
    error_task_1_ohnecu_over_NrParam = np.ones((Nr_parameter_plot))
    error_task_2_ohnecu_over_NrParam = np.ones((Nr_parameter_plot))

    for parameter_iter in range(Nr_parameter_plot):


        load_model_iter = 'model_' + str(int(load_model[-1]) + parameter_iter)
        if load_model[-2] == '1':
            load_model_iter = 'model_1' + str(int(load_model[-1]) + parameter_iter)
        with open('dual-cooperative-semantic-main/results/'+load_directory+"/"+load_model_iter+'/objs.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
            k_1, k_2, k_3, s_1, s_2, s_3, n_x_latent, transmit_CU, centralised_observation, SNR_task_1, SNR_task_2 = pickle.load(f)
        
        #\left(\left(9+1\right)k_{1}+\left(\left(9\cdot k_{1}\right)+1\right)k_{2}\ +2\left(\left(\left(9k_{2}\right)+1\right)k_{3}+l\cdot\left(1+k_{3}*3*3\right)\right)\right)
        n_x_latent = 0
        if centralised_observation == False:
            Nr_of_parameters_withCU[parameter_iter] =  (   (9+1)*k_1+ (9*k_1+1)*k_2 + 2*(9*k_2+1)*k_3 + 2*n_x_latent*(k_3*3*3+1)   )
            Nr_of_parameters_ohneCU[parameter_iter] =  (   2*(9+1)*s_1+ 2*(9*s_1+1)*s_2 + 2*(9*s_2+1)*s_3 + 2*n_x_latent*(s_3*3*3+1)   )
        else:
            Nr_of_parameters_withCU[parameter_iter] = (9+1)*k_1+ (9*k_1+1)*k_2 + 2*(9*k_2+1)*k_3 + 2*n_x_latent*(k_3*3*3+1) # 7*7
            Nr_of_parameters_ohneCU[parameter_iter] = 2*(9+1)*s_1+ 2*(9*s_1+1)*s_2 + 2*(9*s_2+1)*s_3 + 2*n_x_latent*(s_3*3*3+1)  #7*7


        SNR_save_dir = 'dual-cooperative-semantic-main/results/'+load_directory+"/"+load_model_iter+ '/error_rates_over_SNR' 
        error_dir = 'dual-cooperative-semantic-main/results/'+load_directory+"/"+load_model_iter+'/error_rates'

        avg_error_task_1_mitcu_overSNR = load_and_average(SNR_save_dir, 'error_task_1_iter')
        avg_error_task_2_mitcu_overSNR = load_and_average(SNR_save_dir, 'error_task_2_iter')
        avg_error_task_1_ohnecu_overSNR = load_and_average(SNR_save_dir, 'error_task_1_ohnecu_iter')
        avg_error_task_2_ohnecu_overSNR = load_and_average(SNR_save_dir, 'error_task_2_ohnecu_iter')

        original_SNR_linspace_length = len(SNR_eval_linspace)
        avg_error_task_1_mitcu_overSNR = avg_error_task_1_mitcu_overSNR[-original_SNR_linspace_length:]
        avg_error_task_2_mitcu_overSNR = avg_error_task_2_mitcu_overSNR[-original_SNR_linspace_length:]
        avg_error_task_1_ohnecu_overSNR = avg_error_task_1_ohnecu_overSNR[-original_SNR_linspace_length:]
        avg_error_task_2_ohnecu_overSNR = avg_error_task_2_ohnecu_overSNR[-original_SNR_linspace_length:]
        
        avg_error_task_1 = load_and_average(error_dir,'error_task_1_iter')
        avg_error_task_1_ohnecu = load_and_average(error_dir,'error_task_1_ohnecu_iter')
        avg_error_task_2 = load_and_average(error_dir, 'error_task_2_iter')
        avg_error_task_2_ohnecu = load_and_average(error_dir, 'error_task_2_ohnecu_iter')


        error_task_1_mitcu_over_NrParam[parameter_iter] = avg_error_task_1_mitcu_overSNR[eval_at_SNR_index]
        error_task_2_mitcu_over_NrParam[parameter_iter] = avg_error_task_2_mitcu_overSNR[eval_at_SNR_index_task_2]
        error_task_1_ohnecu_over_NrParam[parameter_iter] = avg_error_task_1_ohnecu_overSNR[eval_at_SNR_index]
        error_task_2_ohnecu_over_NrParam[parameter_iter] = avg_error_task_2_ohnecu_overSNR[eval_at_SNR_index_task_2]


        if SNR == "inf":
            error_task_1_mitcu_over_NrParam[parameter_iter] = avg_error_task_1[-1]
            error_task_2_mitcu_over_NrParam[parameter_iter] = avg_error_task_2[-1]
            error_task_1_ohnecu_over_NrParam[parameter_iter] = avg_error_task_1_ohnecu[-1]
            error_task_2_ohnecu_over_NrParam[parameter_iter] = avg_error_task_2_ohnecu[-1]


    #Plot result over Nr_parameters:
    Nr_SNR_evaluations = 41
    SNR_eval_min = -20
    SNR_eval_max = +20
    SNR_eval_linspace = np.linspace(SNR_eval_min,SNR_eval_max,Nr_SNR_evaluations)
    # Plot of accuracies
    fig = plt.figure()
    plt.semilogy(Nr_of_parameters_withCU,error_task_1_mitcu_over_NrParam, label=label_task1_CUSU_retrainSmallSNRrange, color=color_task1_CUSU_retrainSmallSNRrange, marker=marker_task1_CUSU_retrainSmallSNRrange,markerfacecolor="None")
    plt.semilogy(Nr_of_parameters_withCU,error_task_2_mitcu_over_NrParam, label=label_task2_CUSU_retrainSmallSNRrange, color=color_task2_CUSU_retrainSmallSNRrange, marker=marker_task2_CUSU_retrainSmallSNRrange,markerfacecolor="None")
    plt.semilogy(Nr_of_parameters_ohneCU,error_task_1_ohnecu_over_NrParam, label=label_task1_ohne_CU_retrainSmallSNRrange, color=color_task1_ohne_CU_retrainSmallSNRrange, marker=marker_task1_ohne_CU_retrainSmallSNRrange,markerfacecolor="None")
    plt.semilogy(Nr_of_parameters_ohneCU,error_task_2_ohnecu_over_NrParam, label=label_task2_ohne_CU_retrainSmallSNRrange, color=color_task2_ohne_CU_retrainSmallSNRrange, marker=marker_task2_ohne_CU_retrainSmallSNRrange,markerfacecolor="None")
    #plt.semilogy(Nr_of_parameters_withCU,error_task_1_mitcu_over_NrParam/error_task_1_ohnecu_over_NrParam, label='Task1 difference', color=sand, marker='<',markerfacecolor="None")
    #plt.semilogy(Nr_of_parameters_withCU,error_task_2_mitcu_over_NrParam/error_task_2_ohnecu_over_NrParam, label='Task2 difference', color=green, marker='>',markerfacecolor="None")

    #plt.title('Accuracy of Task Execution (Impact of CU)')
    plt.xlabel('Nr of encoder NN parameters')
    plt.ylabel('Error rate')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    # save plots

    tikzplotlib_fix_ncols(fig) #needed because of bug caused by updated matplotlib that renamed ncols
    tikzplotlib.save('dual-cooperative-semantic-main/plots_CU_tikz/'+load_directory+"/"+load_model+'/NrParameter.tikz', override_externals=True, encoding='utf-8', strict=True)


    plt.title("SNR eval="+str(SNR_eval) + ", Ch.uses="+str(n_x_latent)+"\n"+", central observer:"+str(centralised_observation))
    fig.savefig('dual-cooperative-semantic-main/plots_CU_png/'+load_directory+"/"+load_model+'/NrParameter.png', dpi=fig.dpi)

    plt.show()



if plot_task_1_task_2 == True:
    
    Nr_parameter_plot = 6
    

    
    eval_at_SNR_index_task_1 = 25 # 10 gives 0 SNR; 30 gives 10 SNR
    eval_at_SNR_index_task_2 = 30

    SNR_eval_task_1 = SNR_eval_linspace[eval_at_SNR_index_task_1]
    SNR_eval_task_2 = SNR_eval_linspace[eval_at_SNR_index_task_2]

    Nr_of_parameters_withCU = np.zeros((Nr_parameter_plot))
    Nr_of_parameters_ohneCU = np.zeros((Nr_parameter_plot))

    error_task_1_mitcu_over_NrParam = np.ones((Nr_parameter_plot)) * 10
    error_task_2_mitcu_over_NrParam = np.ones((Nr_parameter_plot)) * 10
    error_task_1_ohnecu_over_NrParam = np.ones((Nr_parameter_plot)) * 10
    error_task_2_ohnecu_over_NrParam = np.ones((Nr_parameter_plot)) * 10

    error_task_1_mitcu_over_NrParam_distributed_perfec_comm = np.ones((Nr_parameter_plot)) * 10 
    error_task_2_mitcu_over_NrParam_distributed_perfec_comm = np.ones((Nr_parameter_plot)) * 10
    error_task_1_ohnecu_over_NrParam_distributed_perfec_comm = np.ones((Nr_parameter_plot)) * 10
    error_task_2_ohnecu_over_NrParam_distributed_perfec_comm = np.ones((Nr_parameter_plot)) * 10

    error_task_1_mitcu_over_NrParam_central = np.zeros((Nr_parameter_plot))
    error_task_2_mitcu_over_NrParam_central = np.zeros((Nr_parameter_plot))
    error_task_1_ohnecu_over_NrParam_central = np.zeros((Nr_parameter_plot))
    error_task_2_ohnecu_over_NrParam_central = np.zeros((Nr_parameter_plot))

    Nr_parameter_plot = 12
    for parameter_iter in range(Nr_parameter_plot):

        if parameter_iter >= 6:
            SNR = "inf"

        load_model_iter = 'model_' + str(int(load_model[-1]) + parameter_iter)
        if load_model[-2] == '1':
            load_model_iter = 'model_1' + str(int(load_model[-1]) + parameter_iter)
        with open('dual-cooperative-semantic-main/results/'+load_directory+"/"+load_model_iter+'/objs.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
            k_1, k_2, k_3, s_1, s_2, s_3, n_x_latent, transmit_CU, centralised_observation, SNR_task_1, SNR_task_2 = pickle.load(f)
        
        #\left(\left(9+1\right)k_{1}+\left(\left(9\cdot k_{1}\right)+1\right)k_{2}\ +2\left(\left(\left(9k_{2}\right)+1\right)k_{3}+l\cdot\left(1+k_{3}*3*3\right)\right)\right)
        n_x_latent = 0

        if parameter_iter <6:
            if centralised_observation == False:
                Nr_of_parameters_withCU[parameter_iter] =  (   (9+1)*k_1+ (9*k_1+1)*k_2 + 2*(9*k_2+1)*k_3 + 2*n_x_latent*(k_3*3*3+1)   )
                Nr_of_parameters_ohneCU[parameter_iter] =  (   2*(9+1)*s_1+ 2*(9*s_1+1)*s_2 + 2*(9*s_2+1)*s_3 + 2*n_x_latent*(s_3*3*3+1)   )
            else:
                Nr_of_parameters_withCU[parameter_iter] = (9+1)*k_1+ (9*k_1+1)*k_2 + 2*(9*k_2+1)*k_3 + 2*n_x_latent*(k_3*3*3+1) # 7*7
                Nr_of_parameters_ohneCU[parameter_iter] = 2*(9+1)*s_1+ 2*(9*s_1+1)*s_2 + 2*(9*s_2+1)*s_3 + 2*n_x_latent*(s_3*3*3+1)  #7*7


        SNR_save_dir = 'dual-cooperative-semantic-main/results/'+load_directory+"/"+load_model_iter+ '/error_rates_over_SNR' 
        error_dir = 'dual-cooperative-semantic-main/results/'+load_directory+"/"+load_model_iter+'/error_rates'

        avg_error_task_1_mitcu_overSNR = load_and_average(SNR_save_dir, 'error_task_1_iter')
        avg_error_task_2_mitcu_overSNR = load_and_average(SNR_save_dir, 'error_task_2_iter')
        avg_error_task_1_ohnecu_overSNR = load_and_average(SNR_save_dir, 'error_task_1_ohnecu_iter')
        avg_error_task_2_ohnecu_overSNR = load_and_average(SNR_save_dir, 'error_task_2_ohnecu_iter')

        original_SNR_linspace_length = len(SNR_eval_linspace)
        avg_error_task_1_mitcu_overSNR = avg_error_task_1_mitcu_overSNR[-original_SNR_linspace_length:]
        avg_error_task_2_mitcu_overSNR = avg_error_task_2_mitcu_overSNR[-original_SNR_linspace_length:]
        avg_error_task_1_ohnecu_overSNR = avg_error_task_1_ohnecu_overSNR[-original_SNR_linspace_length:]
        avg_error_task_2_ohnecu_overSNR = avg_error_task_2_ohnecu_overSNR[-original_SNR_linspace_length:]

        avg_error_task_1 = load_and_average(error_dir,'error_task_1_iter')
        avg_error_task_1_ohnecu = load_and_average(error_dir,'error_task_1_ohnecu_iter')
        avg_error_task_2 = load_and_average(error_dir, 'error_task_2_iter')
        avg_error_task_2_ohnecu = load_and_average(error_dir, 'error_task_2_ohnecu_iter')



        if parameter_iter <6:
            error_task_1_mitcu_over_NrParam[parameter_iter] = avg_error_task_1_mitcu_overSNR[eval_at_SNR_index_task_1]
            error_task_2_mitcu_over_NrParam[parameter_iter] = avg_error_task_2_mitcu_overSNR[eval_at_SNR_index_task_2]
            error_task_1_ohnecu_over_NrParam[parameter_iter] = avg_error_task_1_ohnecu_overSNR[eval_at_SNR_index_task_1]
            error_task_2_ohnecu_over_NrParam[parameter_iter] = avg_error_task_2_ohnecu_overSNR[eval_at_SNR_index_task_2]


        elif parameter_iter >=6:
            error_task_1_mitcu_over_NrParam_distributed_perfec_comm[parameter_iter-6] = avg_error_task_1[-1]
            error_task_2_mitcu_over_NrParam_distributed_perfec_comm[parameter_iter-6] = avg_error_task_2[-1]
            error_task_1_ohnecu_over_NrParam_distributed_perfec_comm[parameter_iter-6] = avg_error_task_1_ohnecu[-1]
            error_task_2_ohnecu_over_NrParam_distributed_perfec_comm[parameter_iter-6] = avg_error_task_2_ohnecu[-1]

        else:
            error_task_1_mitcu_over_NrParam_central[parameter_iter-6] = avg_error_task_1[-1]
            error_task_2_mitcu_over_NrParam_central[parameter_iter-6] = avg_error_task_2[-1]
            error_task_1_ohnecu_over_NrParam_central[parameter_iter-6] = avg_error_task_1_ohnecu[-1]
            error_task_2_ohnecu_over_NrParam_central[parameter_iter-6] = avg_error_task_2_ohnecu[-1]

        
    # temp = error_task_1_ohnecu_over_NrParam_distributed_perfec_comm[2]
    # error_task_1_ohnecu_over_NrParam_distributed_perfec_comm[2] = error_task_1_ohnecu_over_NrParam[2]
    # error_task_1_ohnecu_over_NrParam[2] = temp

    #Plot result over Nr_parameters:


    #Plot erros task 1 and task 2
    fig = plt.figure()
    plt.semilogy(Nr_of_parameters_ohneCU,error_task_1_ohnecu_over_NrParam, label=label_task1_ohne_CU_retrainSmallSNRrange, color=color_task1_ohne_CU_retrainSmallSNRrange, marker=marker_task1_ohne_CU_retrainSmallSNRrange,markerfacecolor="None")
    plt.semilogy(Nr_of_parameters_withCU,error_task_1_mitcu_over_NrParam, label=label_task1_CUSU_retrainSmallSNRrange, color=color_task1_CUSU_retrainSmallSNRrange, marker=marker_task1_CUSU_retrainSmallSNRrange,markerfacecolor="None")
    
    plt.semilogy(Nr_of_parameters_ohneCU,error_task_1_ohnecu_over_NrParam_distributed_perfec_comm, label=label_task1_ohne_CU_noChNoise, color=color_task1_ohne_CU_noChNoise, marker=marker_task1_ohne_CU_noChNoise,markerfacecolor="None")
    plt.semilogy(Nr_of_parameters_withCU,error_task_1_mitcu_over_NrParam_distributed_perfec_comm, label=label_task1_CUSU_noChNoise, color=color_task1_CUSU_noChNoise, marker=marker_task1_CUSU_noChNoise,markerfacecolor="None")
    
    plt.semilogy(Nr_of_parameters_ohneCU,error_task_2_ohnecu_over_NrParam, label=label_task2_ohne_CU_retrainSmallSNRrange, color=color_task2_ohne_CU_retrainSmallSNRrange, marker=marker_task2_ohne_CU_retrainSmallSNRrange,markerfacecolor="None")
    plt.semilogy(Nr_of_parameters_withCU,error_task_2_mitcu_over_NrParam, label=label_task2_CUSU_retrainSmallSNRrange, color=color_task2_CUSU_retrainSmallSNRrange, marker=marker_task2_CUSU_retrainSmallSNRrange,markerfacecolor="None")
    
    plt.semilogy(Nr_of_parameters_ohneCU,error_task_2_ohnecu_over_NrParam_distributed_perfec_comm, label=label_task2_ohne_CU_noChNoise, color=color_task2_ohne_CU_noChNoise, marker=marker_task2_ohne_CU_noChNoise,markerfacecolor="None")
    plt.semilogy(Nr_of_parameters_withCU,error_task_2_mitcu_over_NrParam_distributed_perfec_comm, label=label_task2_CUSU_noChNoise, color=color_task2_CUSU_noChNoise, marker=marker_task2_CUSU_noChNoise,markerfacecolor="None")
    
    #plt.title('Accuracy of Task Execution (Impact of CU)')
    plt.xlabel('Nr of encoder NN parameters')
    plt.ylabel('Error rate')
    plt.legend()
    plt.grid(True)
    #plt.ylim([0, 0.1])
    # save plots

    tikzplotlib_fix_ncols(fig) #needed because of bug caused by updated matplotlib that renamed ncols
    tikzplotlib.save('dual-cooperative-semantic-main/plots_CU_tikz/'+load_directory+"/"+load_model+'/NrParameter_task_1_and_task2.tikz', override_externals=True, encoding='utf-8', strict=True)


    plt.title("SNR eval task 1="+str(SNR_eval_task_1)+  ", Ch.uses="+str(n_x_latent)+"\n"+", central observer:"+str(centralised_observation))
    fig.savefig('dual-cooperative-semantic-main/plots_CU_png/'+load_directory+"/"+load_model+'/NrParameter_task_1_and_task2.png', dpi=fig.dpi)

    plt.show()


    # Plot of errors task 1
    fig = plt.figure()
    plt.semilogy(Nr_of_parameters_ohneCU,error_task_1_ohnecu_over_NrParam, label=label_task1_ohne_CU_retrainSmallSNRrange, color=color_task1_ohne_CU_retrainSmallSNRrange, marker=marker_task1_ohne_CU_retrainSmallSNRrange,markerfacecolor="None")
    plt.semilogy(Nr_of_parameters_withCU,error_task_1_mitcu_over_NrParam, label=label_task1_CUSU_retrainSmallSNRrange, color=color_task1_CUSU_retrainSmallSNRrange, marker=marker_task1_CUSU_retrainSmallSNRrange,markerfacecolor="None")
    
    plt.semilogy(Nr_of_parameters_ohneCU,error_task_1_ohnecu_over_NrParam_distributed_perfec_comm, label=label_task1_ohne_CU_noChNoise, color=color_task1_ohne_CU_noChNoise, marker=marker_task1_ohne_CU_noChNoise,markerfacecolor="None")
    plt.semilogy(Nr_of_parameters_withCU,error_task_1_mitcu_over_NrParam_distributed_perfec_comm, label=label_task1_CUSU_noChNoise, color=color_task1_CUSU_noChNoise, marker=marker_task1_CUSU_noChNoise,markerfacecolor="None")
    
    #plt.semilogy(Nr_of_parameters_withCU,error_task_1_mitcu_over_NrParam_central, label='Central Error (With CU)', color=green, marker='*',markerfacecolor="None")
    #plt.semilogy(Nr_of_parameters_ohneCU,error_task_1_ohnecu_over_NrParam_central, label='Central Error (Without CU)', color=sand, marker='H',markerfacecolor="None")

    #plt.title('Accuracy of Task Execution (Impact of CU)')
    plt.xlabel('Nr of encoder NN parameters')
    plt.ylabel('Error rate')
    plt.legend()
    plt.grid(True)
    #plt.ylim([0, 0.1])
    # save plots

    tikzplotlib_fix_ncols(fig) #needed because of bug caused by updated matplotlib that renamed ncols
    tikzplotlib.save('dual-cooperative-semantic-main/plots_CU_tikz/'+load_directory+"/"+load_model+'/NrParameter_task_1.tikz', override_externals=True, encoding='utf-8', strict=True)


    plt.title("SNR eval task 1="+str(SNR_eval_task_1)+  ", Ch.uses="+str(n_x_latent)+"\n"+", central observer:"+str(centralised_observation))
    fig.savefig('dual-cooperative-semantic-main/plots_CU_png/'+load_directory+"/"+load_model+'/NrParameter_task_1.png', dpi=fig.dpi)

    plt.show()


    # Plot of error task 2
    fig = plt.figure()
    plt.semilogy(Nr_of_parameters_ohneCU,error_task_2_ohnecu_over_NrParam, label=label_task2_ohne_CU_retrainSmallSNRrange, color=color_task2_ohne_CU_retrainSmallSNRrange, marker=marker_task2_ohne_CU_retrainSmallSNRrange,markerfacecolor="None")
    plt.semilogy(Nr_of_parameters_withCU,error_task_2_mitcu_over_NrParam, label=label_task2_CUSU_retrainSmallSNRrange, color=color_task2_CUSU_retrainSmallSNRrange, marker=marker_task2_CUSU_retrainSmallSNRrange,markerfacecolor="None")
    
    plt.semilogy(Nr_of_parameters_ohneCU,error_task_2_ohnecu_over_NrParam_distributed_perfec_comm, label=label_task2_ohne_CU_noChNoise, color=color_task2_ohne_CU_noChNoise, marker=marker_task2_ohne_CU_noChNoise,markerfacecolor="None")
    plt.semilogy(Nr_of_parameters_withCU,error_task_2_mitcu_over_NrParam_distributed_perfec_comm, label=label_task2_CUSU_noChNoise, color=color_task2_CUSU_noChNoise, marker=marker_task2_CUSU_noChNoise,markerfacecolor="None")
    
    #plt.plot(Nr_of_parameters_withCU,error_task_2_mitcu_over_NrParam_central, label='Central Error Task2 (With CU)', color=green, marker='*',markerfacecolor="None")
    #plt.plot(Nr_of_parameters_ohneCU,error_task_2_ohnecu_over_NrParam_central, label='Central Error Task2 (Without CU)', color=sand, marker='H',markerfacecolor="None")

    #plt.title('Accuracy of Task Execution (Impact of CU)')
    plt.xlabel('Nr of encoder NN parameters')
    plt.ylabel('Error rate')
    plt.legend()
    plt.grid(True)
    #plt.ylim(0, 0.14)
    # save plots

    tikzplotlib_fix_ncols(fig) #needed because of bug caused by updated matplotlib that renamed ncols
    tikzplotlib.save('dual-cooperative-semantic-main/plots_CU_tikz/'+load_directory+"/"+load_model+'/NrParameter_task_2.tikz', override_externals=True, encoding='utf-8', strict=True)


    plt.title("SNR eval task 2="+str(SNR_eval_task_2) + ", Ch.uses="+str(n_x_latent)+"\n"+", central observer:"+str(centralised_observation))
    fig.savefig('dual-cooperative-semantic-main/plots_CU_png/'+load_directory+"/"+load_model+'/NrParameter_task_2.png', dpi=fig.dpi)

    plt.show()



def find_index(linspace_vec, value):
    # Find the index of the closest value in the linspace vector
    index = np.argmin(np.abs(linspace_vec - value))
    return index

if plot_SNR_retrain == True:

    Nr_snr_eval_per_retrain = 3
    Nr_of_retrained_models = 11

    Nr_SNR_evaluations = Nr_snr_eval_per_retrain*Nr_of_retrained_models

    Nr_parameter_load = 13
    

    min_snr_eval_retrain = -12
    max_snr_eval_retrain = 20
    SNR_eval_linspace_retrain_plot = np.linspace(min_snr_eval_retrain,max_snr_eval_retrain,Nr_SNR_evaluations)




    error_task_1_mitcu_over_SNR_retrain = np.ones((Nr_SNR_evaluations)) * 10
    error_task_2_mitcu_over_SNR_retrain = np.ones((Nr_SNR_evaluations)) * 10
    error_task_1_ohnecu_over_SNR_retrain = np.ones((Nr_SNR_evaluations)) * 10
    error_task_2_ohnecu_over_SNR_retrain = np.ones((Nr_SNR_evaluations)) * 10
    error_task_1_mitcu_over_SNR_no_retrain = np.ones((Nr_SNR_evaluations)) * 10
    error_task_2_mitcu_over_SNR_no_retrain = np.ones((Nr_SNR_evaluations)) * 10


    for parameter_iter in range(Nr_parameter_load):


        load_model_iter = 'model_' + str(int(load_model[-1]) + parameter_iter)
        if load_model[-2] == '1':
            load_model_iter = 'model_1' + str(int(load_model[-1]) + parameter_iter)
        with open('dual-cooperative-semantic-main/results/'+load_directory+"/"+load_model_iter+'/objs.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
            k_1, k_2, k_3, s_1, s_2, s_3, n_x_latent, transmit_CU, centralised_observation, SNR_task_1, SNR_task_2 = pickle.load(f)
        
        
        SNR_save_dir = 'dual-cooperative-semantic-main/results/'+load_directory+"/"+load_model_iter+ '/error_rates_over_SNR' 
        error_dir = 'dual-cooperative-semantic-main/results/'+load_directory+"/"+load_model_iter+'/error_rates'

        avg_error_task_1_mitcu_overSNR = load_and_average(SNR_save_dir, 'error_task_1_iter')
        avg_error_task_2_mitcu_overSNR = load_and_average(SNR_save_dir, 'error_task_2_iter')
        avg_error_task_1_ohnecu_overSNR = load_and_average(SNR_save_dir, 'error_task_1_ohnecu_iter')
        avg_error_task_2_ohnecu_overSNR = load_and_average(SNR_save_dir, 'error_task_2_ohnecu_iter')

        original_SNR_linspace_length = len(SNR_eval_linspace)
        avg_error_task_1_mitcu_overSNR = avg_error_task_1_mitcu_overSNR[-original_SNR_linspace_length:]
        avg_error_task_2_mitcu_overSNR = avg_error_task_2_mitcu_overSNR[-original_SNR_linspace_length:]
        avg_error_task_1_ohnecu_overSNR = avg_error_task_1_ohnecu_overSNR[-original_SNR_linspace_length:]
        avg_error_task_2_ohnecu_overSNR = avg_error_task_2_ohnecu_overSNR[-original_SNR_linspace_length:]

        avg_error_task_1 = load_and_average(error_dir,'error_task_1_iter')
        avg_error_task_1_ohnecu = load_and_average(error_dir,'error_task_1_ohnecu_iter')
        avg_error_task_2 = load_and_average(error_dir, 'error_task_2_iter')
        avg_error_task_2_ohnecu = load_and_average(error_dir, 'error_task_2_ohnecu_iter')


        index_task_1 = find_index(SNR_eval_linspace,SNR_task_1)
        index_task_2 = find_index(SNR_eval_linspace,SNR_task_2)

        if parameter_iter < Nr_of_retrained_models:
            
            index_min_snr_linspace_conversion = find_index(SNR_eval_linspace,min_snr_eval_retrain)
            index_max_snr_linspace_conversion = find_index(SNR_eval_linspace,max_snr_eval_retrain)

            error_task_1_mitcu_over_SNR_retrain_new = avg_error_task_1_mitcu_overSNR[index_min_snr_linspace_conversion:index_max_snr_linspace_conversion+1]
            error_task_2_mitcu_over_SNR_retrain_new = avg_error_task_2_mitcu_overSNR[index_min_snr_linspace_conversion:index_max_snr_linspace_conversion+1]
            error_task_1_ohnecu_over_SNR_retrain_new = avg_error_task_1_ohnecu_overSNR[index_min_snr_linspace_conversion:index_max_snr_linspace_conversion+1]
            error_task_2_ohnecu_over_SNR_retrain_new = avg_error_task_2_ohnecu_overSNR[index_min_snr_linspace_conversion:index_max_snr_linspace_conversion+1]

            error_task_1_mitcu_over_SNR_retrain = np.min([error_task_1_mitcu_over_SNR_retrain,error_task_1_mitcu_over_SNR_retrain_new],axis=0)
            error_task_2_mitcu_over_SNR_retrain = np.min([error_task_2_mitcu_over_SNR_retrain,error_task_2_mitcu_over_SNR_retrain_new],axis=0)
            error_task_1_ohnecu_over_SNR_retrain = np.min([error_task_1_ohnecu_over_SNR_retrain,error_task_1_ohnecu_over_SNR_retrain_new],axis=0)
            error_task_2_ohnecu_over_SNR_retrain = np.min([error_task_2_ohnecu_over_SNR_retrain,error_task_2_ohnecu_over_SNR_retrain_new],axis=0) 

        elif parameter_iter == Nr_of_retrained_models:
            #only 1 model over all SNR
            index_min_snr_linspace_conversion = find_index(SNR_eval_linspace,min_snr_eval_retrain)
            index_max_snr_linspace_conversion = find_index(SNR_eval_linspace,max_snr_eval_retrain)

            error_task_1_mitcu_over_all_SNR = avg_error_task_1_mitcu_overSNR[index_min_snr_linspace_conversion:index_max_snr_linspace_conversion+1]
            error_task_2_mitcu_over_all_SNR = avg_error_task_2_mitcu_overSNR[index_min_snr_linspace_conversion:index_max_snr_linspace_conversion+1]
            error_task_1_ohnecu_over_all_SNR = avg_error_task_1_ohnecu_overSNR[index_min_snr_linspace_conversion:index_max_snr_linspace_conversion+1]
            error_task_2_ohnecu_over_all_SNR = avg_error_task_2_ohnecu_overSNR[index_min_snr_linspace_conversion:index_max_snr_linspace_conversion+1]

        else:
            for intra_index in range(Nr_snr_eval_per_retrain):

                error_task_1_mitcu_over_SNR_no_retrain[(parameter_iter-Nr_of_retrained_models-1) * Nr_snr_eval_per_retrain + intra_index ] = avg_error_task_1_mitcu_overSNR[index_task_1+intra_index-int(Nr_snr_eval_per_retrain/2)]
                error_task_2_mitcu_over_SNR_no_retrain[(parameter_iter-Nr_of_retrained_models-1) * Nr_snr_eval_per_retrain + intra_index] = avg_error_task_2_mitcu_overSNR[index_task_2+intra_index-int(Nr_snr_eval_per_retrain/2)]

        if load_model_iter  == "model_8":
            #retrained SUCU
            training_over_time_retrain_task1 = avg_error_task_1
            training_over_time_ohnecu_task1 = avg_error_task_1_ohnecu
            training_over_time_retrain_task2 = avg_error_task_2
            training_over_time_ohnecu_task2 = avg_error_task_2_ohnecu

            error_over_SNR_retrain_task1 = avg_error_task_1_mitcu_overSNR
            error_over_SNR_retrain_task2 = avg_error_task_2_mitcu_overSNR
            error_over_SNR_retrain_ohne_cu_task1 = avg_error_task_1_ohnecu_overSNR
            error_over_SNR_retrain_ohne_cu_task2 = avg_error_task_2_ohnecu_overSNR

        if load_model_iter == "model_13":
            #non retrained SUCU
            training_over_time_no_retrain_task1 = avg_error_task_1
            training_over_time_no_retrain_task2 = avg_error_task_2
            
            error_over_SNR_no_retrain_task1 = avg_error_task_1_mitcu_overSNR
            error_over_SNR_no_retrain_task2 = avg_error_task_2_mitcu_overSNR




    # plot or errors over SNR with retrain vs no retrain for Task 1 and Task 2
    fig = plt.figure()
    
    plt.semilogy(SNR_eval_linspace_retrain_plot,error_task_1_ohnecu_over_all_SNR, label=label_task1_ohne_CU_trainFullSNRrange, color=color_task1_ohne_CU_trainFullSNRrange, marker=marker_task1_ohne_CU_trainFullSNRrange,markerfacecolor="None")
    plt.semilogy(SNR_eval_linspace_retrain_plot,error_task_1_mitcu_over_all_SNR, label=label_task1_CUSU_trainFullSNRrange, color=color_task1_CUSU_trainFullSNRrange, marker=marker_task1_CUSU_trainFullSNRrange,markerfacecolor="None")
    
    plt.semilogy(SNR_eval_linspace_retrain_plot,error_task_1_ohnecu_over_SNR_retrain, label=label_task1_ohne_CU_retrainSmallSNRrange, color=color_task1_ohne_CU_retrainSmallSNRrange, marker=marker_task1_ohne_CU_retrainSmallSNRrange,markerfacecolor="None")
    plt.semilogy(SNR_eval_linspace_retrain_plot,error_task_1_mitcu_over_SNR_retrain, label=label_task1_CUSU_retrainSmallSNRrange, color=color_task1_CUSU_retrainSmallSNRrange, marker=marker_task1_CUSU_retrainSmallSNRrange,markerfacecolor="None")
    
    
    plt.semilogy(SNR_eval_linspace_retrain_plot,error_task_2_ohnecu_over_all_SNR, label=label_task2_ohne_CU_trainFullSNRrange, color=color_task2_ohne_CU_trainFullSNRrange, marker=marker_task2_ohne_CU_trainFullSNRrange,markerfacecolor="None")
   
    plt.semilogy(SNR_eval_linspace_retrain_plot,error_task_2_mitcu_over_all_SNR, label=label_task2_CUSU_trainFullSNRrange, color=color_task2_CUSU_trainFullSNRrange, marker=marker_task2_CUSU_trainFullSNRrange,markerfacecolor="None")
    
    plt.semilogy(SNR_eval_linspace_retrain_plot,error_task_2_ohnecu_over_SNR_retrain, label=label_task2_ohne_CU_retrainSmallSNRrange, color=color_task2_ohne_CU_retrainSmallSNRrange, marker=marker_task2_ohne_CU_retrainSmallSNRrange,markerfacecolor="None")

    plt.semilogy(SNR_eval_linspace_retrain_plot,error_task_2_mitcu_over_SNR_retrain, label=label_task2_CUSU_retrainSmallSNRrange, color=color_task2_CUSU_retrainSmallSNRrange, marker=marker_task2_CUSU_retrainSmallSNRrange,markerfacecolor="None")
    
    #plt.title('Accuracy of Task Execution (Impact of CU)')
    plt.xlabel('SNR')
    plt.ylabel('Error rate')
    plt.legend()
    plt.grid(True)
    plt.ylim([0, 1])
    # save plots

    tikzplotlib_fix_ncols(fig) #needed because of bug caused by updated matplotlib that renamed ncols
    tikzplotlib.save('dual-cooperative-semantic-main/plots_CU_tikz/'+load_directory+"/"+load_model+'/Over_SNR_with_SU_retrain.tikz', override_externals=True, encoding='utf-8', strict=True)

    plt.title("plot over SNR with SU retrain for specific snr ranges")
    fig.savefig('dual-cooperative-semantic-main/plots_CU_png/'+load_directory+"/"+load_model+'/Over_SNR_with_SU_retrain.png', dpi=fig.dpi)

    plt.show()


    
    # plot or errors over SNR with retrain vs no retrain for Task 1 only
    fig = plt.figure()
    plt.semilogy(SNR_eval_linspace_retrain_plot,error_task_1_mitcu_over_SNR_retrain, label=label_task1_CUSU_retrainSmallSNRrange, color=color_task1_CUSU_retrainSmallSNRrange, marker=marker_task1_CUSU_retrainSmallSNRrange,markerfacecolor="None")
    plt.semilogy(SNR_eval_linspace_retrain_plot,error_task_1_ohnecu_over_SNR_retrain, label=label_task1_ohne_CU_retrainSmallSNRrange, color=color_task1_ohne_CU_retrainSmallSNRrange, marker=marker_task1_ohne_CU_retrainSmallSNRrange,markerfacecolor="None")

    plt.semilogy(SNR_eval_linspace_retrain_plot,error_task_1_mitcu_over_all_SNR, label=label_task1_CUSU_trainFullSNRrange, color=color_task1_CUSU_trainFullSNRrange, marker=marker_task1_CUSU_trainFullSNRrange,markerfacecolor="None")
    plt.semilogy(SNR_eval_linspace_retrain_plot,error_task_1_ohnecu_over_all_SNR, label=label_task1_ohne_CU_trainFullSNRrange, color=color_task1_ohne_CU_trainFullSNRrange, marker=marker_task1_ohne_CU_trainFullSNRrange,markerfacecolor="None")

    #plt.title('Accuracy of Task Execution (Impact of CU)')
    plt.xlabel('SNR')
    plt.ylabel('Error rate')
    plt.legend()
    plt.grid(True)
    plt.ylim([0, 1])
    # save plots

    tikzplotlib_fix_ncols(fig) #needed because of bug caused by updated matplotlib that renamed ncols
    tikzplotlib.save('dual-cooperative-semantic-main/plots_CU_tikz/'+load_directory+"/"+load_model+'/Over_SNR_with_SU_retrain_task1.tikz', override_externals=True, encoding='utf-8', strict=True)

    plt.title("Task1: plot over SNR with SU retrain for specific snr ranges")
    fig.savefig('dual-cooperative-semantic-main/plots_CU_png/'+load_directory+"/"+load_model+'/Over_SNR_with_SU_retrain_task1.png', dpi=fig.dpi)

    plt.show()


    # plot or errors over SNR with retrain vs no retrain for Task 2 only
    fig = plt.figure()

    plt.semilogy(SNR_eval_linspace_retrain_plot,error_task_2_mitcu_over_SNR_retrain, label=label_task2_CUSU_retrainSmallSNRrange, color=color_task2_CUSU_retrainSmallSNRrange, marker=marker_task2_CUSU_retrainSmallSNRrange,markerfacecolor="None")
    plt.semilogy(SNR_eval_linspace_retrain_plot,error_task_2_ohnecu_over_SNR_retrain, label=label_task2_ohne_CU_retrainSmallSNRrange, color=color_task2_ohne_CU_retrainSmallSNRrange, marker=marker_task2_ohne_CU_retrainSmallSNRrange,markerfacecolor="None")

    plt.semilogy(SNR_eval_linspace_retrain_plot,error_task_2_mitcu_over_all_SNR, label=label_task2_CUSU_trainFullSNRrange, color=color_task2_CUSU_trainFullSNRrange, marker=marker_task2_CUSU_trainFullSNRrange,markerfacecolor="None")
    plt.semilogy(SNR_eval_linspace_retrain_plot,error_task_2_ohnecu_over_all_SNR, label=label_task2_ohne_CU_trainFullSNRrange, color=color_task2_ohne_CU_trainFullSNRrange, marker=marker_task2_ohne_CU_trainFullSNRrange,markerfacecolor="None")

    #plt.title('Accuracy of Task Execution (Impact of CU)')
    plt.xlabel('SNR')
    plt.ylabel('Error rate')
    plt.legend()
    plt.grid(True)
    plt.ylim([0, 1])
    # save plots

    tikzplotlib_fix_ncols(fig) #needed because of bug caused by updated matplotlib that renamed ncols
    tikzplotlib.save('dual-cooperative-semantic-main/plots_CU_tikz/'+load_directory+"/"+load_model+'/Over_SNR_with_SU_retrain_task2.tikz', override_externals=True, encoding='utf-8', strict=True)

    plt.title("Task 2: plot over SNR with SU retrain for specific snr ranges ")
    fig.savefig('dual-cooperative-semantic-main/plots_CU_png/'+load_directory+"/"+load_model+'/Over_SNR_with_SU_retrain_task2.png', dpi=fig.dpi)

    plt.show()






    # plot or errors training for Retrain SU and training CU and SU for small SNR range
    
    if load_directory == "run_32_SNR":
        error_dir = 'dual-cooperative-semantic-main/results/'+"run_32_pretrain_CU_for_perfect_ch"+"/"+"model_1"+'/error_rates'
        avg_error_task_1_pretrain_CU_for_perfect_ch = load_and_average(error_dir,'error_task_1_iter')
        avg_error_task_1_ohnecu_pretrain_CU_for_perfect_ch = load_and_average(error_dir,'error_task_1_ohnecu_iter')
        avg_error_task_2_pretrain_CU_for_perfect_ch = load_and_average(error_dir, 'error_task_2_iter')
        avg_error_task_2_ohnecu_pretrain_CU_for_perfect_ch = load_and_average(error_dir, 'error_task_2_ohnecu_iter')
    
    else:
        avg_error_task_1_pretrain_CU_for_perfect_ch = training_over_time_retrain_task1 +0.01
        avg_error_task_2_pretrain_CU_for_perfect_ch = training_over_time_retrain_task1 +0.01


    fig = plt.figure()
    plt.semilogy(training_over_time_ohnecu_task1, label=label_task1_ohne_CU_retrainSmallSNRrange, color=color_task1_ohne_CU_retrainSmallSNRrange, markerfacecolor="None") #marker=marker_task1_ohne_CU_retrainSmallSNRrange,
    
    plt.semilogy(avg_error_task_1_pretrain_CU_for_perfect_ch, label=label_task1_CUSU_pretrain_perfect_ch, color=color_task1_CUSU_pretrain_perfect_ch, markerfacecolor="None")
    plt.semilogy(training_over_time_no_retrain_task1, label=label_task1_CUSU_TrainSmallSNRrange, color=color_task1_CUSU_TrainSmallSNRrange, markerfacecolor="None") #marker=marker_task1_CUSU_TrainSmallSNRrange,

    plt.semilogy(training_over_time_retrain_task1, label=label_task1_CUSU_retrainSmallSNRrange, color=color_task1_CUSU_retrainSmallSNRrange ,markerfacecolor="None") #marker=marker_task1_CUSU_retrainSmallSNRrange,
    

    plt.semilogy(training_over_time_ohnecu_task2, label=label_task2_ohne_CU_retrainSmallSNRrange, color=color_task2_ohne_CU_retrainSmallSNRrange, markerfacecolor="None") #marker=marker_task2_ohne_CU_retrainSmallSNRrange,
    
    plt.semilogy(avg_error_task_2_pretrain_CU_for_perfect_ch, label=label_task2_CUSU_pretrain_perfect_ch, color=color_task2_CUSU_pretrain_perfect_ch, markerfacecolor="None")
    plt.semilogy(training_over_time_no_retrain_task2, label=label_task2_CUSU_TrainSmallSNRrange, color=color_task2_CUSU_TrainSmallSNRrange, markerfacecolor="None") #marker=marker_task2_CUSU_TrainSmallSNRrange,

    plt.semilogy(training_over_time_retrain_task2, label=label_task2_CUSU_retrainSmallSNRrange, color=color_task2_CUSU_retrainSmallSNRrange, markerfacecolor="None") #marker=marker_task2_CUSU_retrainSmallSNRrange,
    
   
    #plt.title('Accuracy of Task Execution (Impact of CU)')
    plt.xlabel('Epoch')
    plt.ylabel('Error rate')
    plt.legend()
    plt.grid(True)
    plt.ylim([0, 1])
    # save plots

    tikzplotlib_fix_ncols(fig) #needed because of bug caused by updated matplotlib that renamed ncols
    tikzplotlib.save('dual-cooperative-semantic-main/plots_CU_tikz/'+load_directory+"/"+load_model+'/Over_Training_Iteration_with_SU_retrain_task1and2.tikz', override_externals=True, encoding='utf-8', strict=True)


    plt.title("plot over Training iteration with SU retrain for specific snr ranges")
    fig.savefig('dual-cooperative-semantic-main/plots_CU_png/'+load_directory+"/"+load_model+'/Over_Training_Iteration_with_SU_retrain_task1and2.png', dpi=fig.dpi)

    plt.show()

    fig = plt.figure()
    plt.semilogy(training_over_time_retrain_task1, label=label_task1_CUSU_retrainSmallSNRrange, color=color_task1_CUSU_retrainSmallSNRrange ,markerfacecolor="None") #marker=marker_task1_CUSU_retrainSmallSNRrange,
    plt.semilogy(training_over_time_ohnecu_task1, label=label_task1_ohne_CU_retrainSmallSNRrange, color=color_task1_ohne_CU_retrainSmallSNRrange, markerfacecolor="None") #marker=marker_task1_ohne_CU_retrainSmallSNRrange,
    plt.semilogy(training_over_time_no_retrain_task1, label=label_task1_CUSU_TrainSmallSNRrange, color=color_task1_CUSU_TrainSmallSNRrange, markerfacecolor="None") #marker=marker_task1_CUSU_TrainSmallSNRrange,

    plt.semilogy(avg_error_task_1_pretrain_CU_for_perfect_ch, label=label_task1_CUSU_pretrain_perfect_ch, color=color_task1_CUSU_pretrain_perfect_ch, markerfacecolor="None")
    

   
    #plt.title('Accuracy of Task Execution (Impact of CU)')
    plt.xlabel('Epoch')
    plt.ylabel('Error rate')
    plt.legend()
    plt.grid(True)
    plt.ylim([0, 1])
    # save plots

    tikzplotlib_fix_ncols(fig) #needed because of bug caused by updated matplotlib that renamed ncols
    tikzplotlib.save('dual-cooperative-semantic-main/plots_CU_tikz/'+load_directory+"/"+load_model+'/Over_Training_Iteration_with_SU_retrain_task1.tikz', override_externals=True, encoding='utf-8', strict=True)


    plt.title("plot over Training iteration with SU retrain for specific snr ranges")
    fig.savefig('dual-cooperative-semantic-main/plots_CU_png/'+load_directory+"/"+load_model+'/Over_Training_Iteration_with_SU_retrain_task1.png', dpi=fig.dpi)

    plt.show()

    fig = plt.figure()
    

    plt.semilogy(training_over_time_ohnecu_task2, label=label_task2_ohne_CU_retrainSmallSNRrange, color=color_task2_ohne_CU_retrainSmallSNRrange, markerfacecolor="None") #marker=marker_task2_ohne_CU_retrainSmallSNRrange,
    plt.semilogy(training_over_time_no_retrain_task2, label=label_task2_CUSU_TrainSmallSNRrange, color=color_task2_CUSU_TrainSmallSNRrange, markerfacecolor="None") #marker=marker_task2_CUSU_TrainSmallSNRrange,

    plt.semilogy(avg_error_task_2_pretrain_CU_for_perfect_ch, label=label_task2_CUSU_pretrain_perfect_ch, color=color_task2_CUSU_pretrain_perfect_ch, markerfacecolor="None")
    plt.semilogy(training_over_time_retrain_task2, label=label_task2_CUSU_retrainSmallSNRrange, color=color_task2_CUSU_retrainSmallSNRrange, markerfacecolor="None") #marker=marker_task2_CUSU_retrainSmallSNRrange,
    
    #plt.title('Accuracy of Task Execution (Impact of CU)')
    plt.xlabel('Epoch')
    plt.ylabel('Error rate')
    plt.legend()
    plt.grid(True)
    plt.ylim([0, 1])
    # save plots

    tikzplotlib_fix_ncols(fig) #needed because of bug caused by updated matplotlib that renamed ncols
    tikzplotlib.save('dual-cooperative-semantic-main/plots_CU_tikz/'+load_directory+"/"+load_model+'/Over_Training_Iteration_with_SU_retrain_task2.tikz', override_externals=True, encoding='utf-8', strict=True)


    plt.title("plot over Training iteration with SU retrain for specific snr ranges")
    fig.savefig('dual-cooperative-semantic-main/plots_CU_png/'+load_directory+"/"+load_model+'/Over_Training_Iteration_with_SU_retrain_task2.png', dpi=fig.dpi)

    plt.show()