### Import Packages
import numpy as np
from scipy import signal
from scipy.signal import resample_poly, firwin, bilinear, fftconvolve, lfilter
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

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

def process_samples(x, samples_rate, center_freq):
    N = len(x)

    # FM Demodulation
    x = 0.5 * np.angle(x[0:-1] * np.conj(x[1:]))

    # Perform frequency shift to put the RDS signal at DC (0 Hz)
    fo = -57e3 
    t = np.arange(N)/sample_rate
    x = x * np.exp(2j*np.pi*fo*t)
    return x

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
    remove_images()
    
    
    x = np.fromfile(file, dtype=np.complex64)

    # perform processing
    x =  process_samples(x, sample_rate, center_freq)

    # TEST AREA
    spectrum(x, sample_rate)
    plt.savefig("images/psd.png")

if __name__ == "__main__":
    main()
