### Import Packages
import numpy as np
from scipy import signal
from scipy.signal import resample_poly, firwin, bilinear, fftconvolve, lfilter
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import time

# clear out everything in images folder
def remove_images():
    path = Path('images')
    for file in path.iterdir():
        file.unlink()

def create_args():
    parser = argparse.ArgumentParser(
            prog="RDS Decoder",
            description="Takes a given IQ file from FM Radio broadcasts, and decodes the RDS information")
    
    parser.add_argument("filename")
    parser.add_argument('-s', '--sample_rate')
    parser.add_argument('-f', '--center_frequency')
    parser.add_argument('-u', '--units')
    
    args = parser.parse_args()

    return args

# Get input paramaters
args = create_args()
units = args.units
if units == "k":
    scaling_factor = 1e3
elif units == "m":
    scaling_factor = 1e6
else:
    scaling_factor = 1
sample_rate = scaling_factor*float(args.sample_rate)
center_freq = float(args.center_frequency)
file = args.filename



def set_sample_rate(decimate_ratio, orig_sample_rate, cutoff):
    temp_rate = orig_sample_rate / decimate_ratio
    downsample = temp_rate / 1e3
    rds_bpsk = 1187.5
    upsample = 0
    while True:
        upsample+=1
        temp_rate = upsample * rds_bpsk
        if temp_rate%rds_bpsk==0.0 and temp_rate>cutoff*2 and temp_rate%2==0:
            break
    return temp_rate/1e3, downsample, temp_rate 

def mueller_muller_sync(samples, interp, coeff, sps):
    mu = 0 # initial estimate of sample phase

    # interpolate the input signal (add detail)
    samples_interp = resample_poly(samples, interp, 1)

    # initialize output
    out = np.zeros(len(samples)+10, dtype=np.complex64)
    out_rail = np.zeros(len(samples)+10, dtype=np.complex64)

    # input and output index
    i_in = 0
    i_out = 2

    while i_out < len(samples) and i_in + interp < len(samples):
        out[i_out] = samples_interp[i_in*interp + int(mu*interp)]
        out_rail[i_out] = int(np.real(out[i_out])>0) + 1j*int(np.imag(out[i_out])>0)

        x = (out_rail[i_out]-out_rail[i_out-2])*np.conj(out[i_out-1])
        y = (out[i_out]-out[i_out-2])*np.conj(out_rail[i_out])

        mm_val = np.real(x-y)
        mu += sps +coeff*mm_val
        i_in += int(np.floor(mu))
        mu = mu - np.floor(mu)
        i_out += 1
    out = out[2:i_out]
    return out

def costas_loop(x, alpha, beta):
    N = len(x)
    phase = 0
    freq = 0

    out = np.zeros(N, dtype=np.complex64)
    freq_log = []

    for i in range(N):
        out[i] = x[i] * np.exp(-1j*phase)
        error = np.real(out[i]*np.imag(out[i]))

        # recalc phase and offet
        freq += (beta*error)
        freq_log.append(freq*sample_rate/(2*np.pi))
        phase += freq + (alpha*error)

        # adjust phase between 0 and 2pi
        while phase >= 2*np.pi:
            phase -= 2*np.pi
        while phase < 0:
            phase += 2*np.pi

    return out, freq_log

def process_samples(x, sample_rate, center_freq):
    # FM Demodulation
    x = 0.5 * np.angle(x[0:-1] * np.conj(x[1:]))

    # Perform frequency shift to put the RDS signal at DC (0 Hz)
    fo = -57e3 
    t = np.arange(len(x))/sample_rate
    x = x * np.exp(2j*np.pi*fo*t)

    # Filter to Isolate RDS signal
    # low-pass
    cutoff = 7.5e3
    taps = firwin(numtaps=53, cutoff=cutoff, fs=sample_rate)
    x = fftconvolve(x, taps, 'valid')

    # Decimate, resample, update sample rate
    decimate_ratio = 10
    x = x[::decimate_ratio]
    upsample, downsample, sample_rate = set_sample_rate(decimate_ratio, sample_rate, cutoff)
    x = resample_poly(x, upsample, downsample)

    # time synchronization
    x = mueller_muller_sync(x, 32, 0.01, 16)

    # fine frequency sync
    x = costas_loop(x, 8.0, 0.02)

    return x

def decode_bpsk(samples):
    bits = (np.real(samples)>0).astype(int) # decode to 1s and 0s
    bits = (bits[1:] - bits[0:-1]) % 2
    bits = bits.astype(np.uint8) # needs to be uint8 for RDS decoder
    return bits

def spectrum(samples, sample_rate):
    y = np.abs(np.fft.fft(samples, 2**14))**2 / (len(samples)*sample_rate)
    y = 10*np.log10(y)
    y = np.fft.fftshift(y)
    
    t = np.linspace(-sample_rate/2, sample_rate/2, len(y))
    
    
    plt.figure(figsize=(12, 4))
    plt.plot(t/scaling_factor, y)
    plt.xlim(0, sample_rate/scaling_factor/2)
    plt.grid()
    
def main():
    # clear out images directory
    #remove_images()
    
    
    x = np.fromfile(file, dtype=np.complex64)
    x_processed = process_samples(x, sample_rate, center_freq)
    bits = decode_bpsk(x)
    print(bits[:10])

    # timing test block
    test = False
    test_num = 10
    if test:
        process_time_results = []
        perf_counter_results = []
        for i in range(test_num):
            t1 = time.process_time()
            t2 = time.perf_counter()
            x1 =  process_samples(x, sample_rate, center_freq)
            elapsed2 = time.perf_counter() - t2
            elapsed1 =  time.process_time() - t1
            process_time_results.append(elapsed1)
            perf_counter_results.append(elapsed2)

        plt.figure()
        plt.plot(process_time_results, '.-')
        plt.plot(perf_counter_results, '.-')
        plt.legend(["process_time()", "perf_counter()"])
        plt.xlabel("Test #")
        plt.ylabel("Time (ns)")
        plt.grid()
        plt.savefig("images/time.png")
        print(f"Time test completed, average time: {np.mean(process_time_results)} s.")


    # fourier transform
    #spectrum(x, sample_rate)
    #plt.savefig("images/psd.png")

if __name__ == "__main__":
    main()
