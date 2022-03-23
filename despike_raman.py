import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

def despiking_algorithm(x, m = None, threshold = None, opt_threshold = False):
    '''
    Spike removal function proposed by Darren A. Whitaker and Kevin Hayes
    Input:
        x: vector or matrix of the numpy.ndarray type containing the Raman spectra to identify the spikes, in 
	the case of a matrix, the samples should be arranged along the lines
        m: number of points (2*m) around the spike to perform the interpolation that will replace the spike value
        threshold: decision threshold value in z-score space to identify possible spikes. When optimizing this 
	parameter, it is recommended to plot the values absolute z-score obtained in the modified_z_score function.
        opt_threshold: boolean, when True, a threshold will be optimized and the function will return its value.
    Output:
        x_out: vector numpy.ndarray containing the corrected spectra, when opt_threshold = False
        threshold: optimized threshold value, when opt_threshold = True
    Examples:
	When the threshold is not defined 
	threshold = despiking_algorithm(x, opt_threshold = True)
	when the threshold has already been set 
	x_out = despiking_algorithm(x,m=4,threshold=7)
    Sandro K. Otani 12/2021
    Reference:
        https://doi.org/10.1016/j.chemolab.2018.06.009
        
    '''  
    try:
        x.shape[1]
        n = x.shape[0]
    except IndexError:
        n = 1
    z_score = []
    if n>1:
        for i in range(n):
            z_score.append(np.abs(modified_z_score(x[i])))
        z_score = np.array(z_score)
    else:
        z_score = np.abs(modified_z_score(x))
    if opt_threshold:
        fig, axs = plt.subplots(2, figsize=(10,10))
        axs[1].plot(z_score.T)
        threshold = 7
        axs[1].axhline(threshold, linewidth=2, color='b',label='Standard threshold = 7')
        axs[1].set_title('z-score plot')
        axs[1].legend()
        axs[0].plot(x.T)
        axs[0].set_title('Original data')
        plt.rcParams.update({'font.size': 20})
        plt.show()
        
        quest_0 = verify_str('Modify the threshold? (y/n)',['y','n'])
        while quest_0 == 'y':
            clear_output(wait=True)
            threshold = handle_type_error('float','Choose a threshold value:',)
            fig, axs = plt.subplots(2, figsize=(10,10))
            axs[1].plot(z_score.T)
            axs[1].axhline(threshold,linewidth=2, color='b',label=f'New threshold = {threshold}')
            axs[1].set_title('Z-score plot')
            axs[1].legend()
            axs[0].plot(x.T)
            axs[0].set_title('Original data')
            plt.rcParams.update({'font.size': 20})
            plt.show()
            quest_0 = verify_str('Choose another threshold value? (y/n)',['y','n'])
        
        return threshold
    else:
        spikes = z_score > threshold
        if n > 1:
            x_out = np.zeros((x.shape[0],x.shape[1]))
            for i in range(n):
                x_out[i,:] = fixer(x[i],m,spikes[i])
        else:
            x_out = fixer(x,m,spikes)
        return x_out
    
def modified_z_score(x):
    '''
    Function to calculate z-score values
    This transformation of the original data into the z-score space allows identify spikes in Raman spectra and 
    set a threshold (threshold) to perform the removal.
    Input:
        x: vector or matrix of the numpy.ndarray type containing the Raman spectra to identify the spikes, in 
	the case of a matrix, the samples should be arranged along the lines
    Output:
        abs_z_score: vector or matrix of type numpy.ndarray containing calculated values of z-score
    '''
    intensity = np.diff(x)
    median_int = np.median(intensity)
    mad_int = np.median([np.abs(intensity - median_int)])
    modified_z_scores = 0.6745 * (intensity - median_int) / mad_int #0.6745 is the 0.75th quartile of the standard normal distribution
    return modified_z_scores

def fixer(x,m,spikes):
    '''
    Function to remove spikes
    Input:
    	x: vector of the numpy.ndarray type containing the Raman spectra
        m: number of points (2*m) around the spike to perform the interpolation that will replace the spike value
        spikes: vector of type numpy.ndarray containing the decision (np.abs(z_score(x)) > threshold)
    Output:
        x_out: vector numpy.ndarray containing the corrected Raman spectra
    '''
    x_out = x.copy() 
    for i in np.arange(len(spikes)):
        if spikes[i] != 0: # If we have an spike in position i
            if i-m<0: # case 1: peak is at the beginning of the spectrum
                w = np.arange(i,i+1+m) # we select m points after our spike       
            elif i+m>len(spikes)-1: # case 2: peak is at the end of the spectrum
                w = np.arange(i-m,i+1) # we select m points before our spike
            else: # case 3: peak is at the middle of the spectrum 
                w = np.arange(i-m,i+1+m) # we select 2m points around our spike
            w2 = w[spikes[w] == 0] 

            x_out[i] = np.mean(x[w2]) 
    return x_out


def handle_type_error(type_, text, resultado = []):
    '''
    Function that prompts the user to enter an int or float number type and returns the converted value
    Input:
    	type: string that defines the type of value to be accepted ("int" or "float")
    	text: string containing the text to be displayed
    Output 
    	resultado: value (int or float) entered by user
    '''
    if type_ == 'int':
        try:
            resultado = int(input(f'{text}'))
        except ValueError:
            resultado = handle_type_error(type_, text, resultado)
        finally:
            return resultado
    if type_ == 'float':
        try:
            resultado = float(input(f'{text}'))
        except ValueError:
            resultado = handle_type_error(type_, text, resultado)
        finally:
            return resultado
def verify_str(text,list_):
    '''
    Function that asks the user to enter a string that must be contained in a predefined list and returns the entered string
    Input:
    	text: string containing the text to be displayed
    	list_: predefined list containing the allowed answers
    Output:
    	value: string allowed entered by user
    '''
    value = input(text).lower()
    if value in list_:
        return value
    else:
        return verify_str(text,list_)
