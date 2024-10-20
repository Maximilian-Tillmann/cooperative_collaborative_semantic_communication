import torch
import numpy as np
import random
import os
    

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    #os.environ["PYTHONHASHSEED"] = str(seed)
    #print(f"Random seed set as {seed}")


def awgn_channel(SNR,x, eval_specific_SNR=False,SNR_train_range=0,iteration=None):
    #SNR = SNR + 10*np.log10(1/4) #shift snr by factor 1/4 to have that SNR is total combined power of distributed/central system


    if eval_specific_SNR == False:
        SNR = SNR + torch.rand_like(x[:,0])*SNR_train_range-SNR_train_range/2
        sigma_n_squared = 10**(-SNR/10)
        sigma = torch.sqrt(sigma_n_squared)
        noise = torch.randn_like(x) * torch.unsqueeze(sigma,dim=1)
    else:
        sigma_n_squared = 10**(-SNR/10)
        sigma = torch.sqrt(torch.tensor(sigma_n_squared)) #convert to torch tensor
        
        # for evaluation: generate noise to given RNG seed:
        np.random.seed(iteration)
        noise_np = np.random.randn(*x.shape)
        noise_np = noise_np.astype("float32")
        noise = torch.from_numpy(noise_np)* sigma
        noise = noise
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        noise = noise.to(device)
        #noise = torch.randn_like(x) * sigma

        #reset np.random seed
        np.random.seed(None)
    return x + noise 


def reparameterize(mu, ln_var):
    std = torch.exp(0.5 * ln_var)
    eps = torch.randn_like(std)
    c = mu + std * eps * 0
    return c