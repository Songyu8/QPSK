import torch
import numpy as np
import math
import torch.nn as nn
import torchvision.transforms as transforms
import time


# Define QPSK symbols
qpsk_symbols = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]])


# Define modulation function
def qpsk_modulation(bits):
    # Reshape input bits into complex symbols
    bits = bits.reshape(-1, 2)

    

    indices = torch.sum(bits * torch.tensor([2, 1]).reshape(1, -1), axis=1)
    indices = indices.long()

   
    # Map indices to symbols
    symbols = qpsk_symbols[indices]
    return symbols

# Define demodulation function
def qpsk_demodulation(symbols):
    # Calculate distance to each possible symbol
    distances = torch.sum((symbols.unsqueeze(1) - qpsk_symbols.unsqueeze(0))**2, axis=2)
    # Find index of closest symbol
    indices = torch.argmin(distances, axis=1)
    # Convert indices to bits
    bits = torch.stack([torch.tensor([i >> 1, i & 1]) for i in indices])
    return bits.reshape(-1)

# Define function to add AWGN
def add_awgn(symbols, snr_db):
    # Calculate SNR in linear scale
    snr = 10**(snr_db / 10)
    # Calculate noise power
    symbol_energy = torch.mean(torch.sum(torch.abs(symbols.float())**2, axis=1))
    noise_power = symbol_energy / snr
    # Generate noise
    noise = torch.randn_like(symbols.float()) * math.sqrt(noise_power)
    # Add noise to symbols
    noisy_symbols = symbols + noise
    return noisy_symbols


# Calculate BER
def calculate_ber(original_bits, received_bits):
    # Count number of bit errors
    errors = torch.sum(original_bits != received_bits)
    # Calculate BER
    ber = errors.float() / original_bits.shape[0]
    return ber.item()


def num2bit(n, bit_width=8):
    """Converts an integer to its binary representation as a tensor."""

    # Convert the integer to a binary string with leading zeros
    Quantbit=n.reshape(n.numel())

    starttime=time.time()

    

  
    binary_strings = [format(val.item(), f'0{bit_width}b') for val in Quantbit]
    print(binary_strings)

    

 

    
    result_tensor = torch.as_tensor([list(map(float, i)) for i in binary_strings])

    endtime=time.time()
    eendtime=endtime-starttime

    print(result_tensor)

    print("程序执行时间：", eendtime, "秒")

    
    return result_tensor

def bit2num(bit_tensor):
    """Converts an 8-bit binary tensor to its corresponding integer value."""

    # Convert the tensor to a binary string
    binary_string = ''.join([str(int(bit)) for bit in bit_tensor])

    # Convert the binary string to an integer
    integer_value = int(binary_string, 2)

    return integer_value



# Define function to quantize analog signal to digital signal
# def quantize_analog_signal(analog_signal, num_bits):
#     # Calculate quantization step size
#     max_value = torch.max(torch.abs(analog_signal))
#     qstep = (2 * max_value) / (2**num_bits - 1)
#     # Quantize analog signal to nearest integer
#     quantized_signal = torch.round(analog_signal / qstep)
#     quantized_signal = quantized_signal.int()
#     return quantized_signal

def quantize_analog_signal(analog_signal, num_bits):    
    # Quantize analog signal to nearest integer    
    quantized_signal = torch.round(analog_signal*(2**num_bits-1))    
    quantized_signal = quantized_signal.int()   
    return quantized_signal

# Define function to dequantize digital signal to analog signal
# def dequantize_digital_signal(digital_signal, num_bits):
#     # Calculate quantization step size
#     max_value = torch.max(torch.abs(digital_signal))
#     qstep = (2 * max_value) / (2**num_bits - 1)
#     # Dequantize digital signal to analog signal
#     analog_signal = digital_signal * qstep
#     return analog_signal

def dequantize_digital_signal(digital_signal, num_bits):    
    # Dequantize digital signal to analog signal    
    analog_signal = digital_signal / (2**num_bits-1)    
    return analog_signal



criterion = nn.MSELoss()



# Test modulation and demodulation functions
if __name__ == '__main__':
    start_time = time.time()
    ADCbit = 8
    analog_signal = torch.rand((12,6))
    shape=analog_signal.size()
    
    
    Quantbit = quantize_analog_signal(analog_signal, ADCbit)
    
    Quantbit_list = (num2bit(Quantbit, ADCbit))

    

    symbols = qpsk_modulation(Quantbit_list)
    noised_symbols = add_awgn(symbols, snr_db=10)

    recovered_bits = qpsk_demodulation(noised_symbols)

   

    recovered_bits = recovered_bits.split(ADCbit)

    recovered_bits = torch.stack(recovered_bits)

    

    BER = calculate_ber(Quantbit_list,recovered_bits)

    Quantbit_Est = [bit2num(bit_tensor) for bit_tensor in recovered_bits]
    Quantbit_Est = torch.tensor(Quantbit_Est)

    
    analog_signal_Est = dequantize_digital_signal(Quantbit_Est, ADCbit)

    
    Quantbit_1=Quantbit.reshape(Quantbit.numel())
    MSE_1 = criterion(Quantbit_1.float() / (2**ADCbit-1), Quantbit_Est.float() / (2**ADCbit-1))
    analog_signal_1=analog_signal.reshape(analog_signal.numel())
    MSE_2 = criterion(analog_signal_1, analog_signal_Est )
    end_time = time.time()
    execution_time = end_time - start_time
    print('BER:', BER, 'MSE_1', MSE_1, 'MSE_2', MSE_2)
    print("程序执行时间：", execution_time, "秒")
