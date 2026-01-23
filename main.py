### Import Packages
import numpy as np
from scipy import signal
from scipy.signal import resample_poly, firwin, bilinear, fftconvolve, lfilter
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


# read in the signal
#x = np.fromfile('drive/MyDrive/Digital Signal Processing/fm_rds_250k_1Msamples.iq', dtype=np.complex64)
#sample_rate = 250e3 # 250 kHz sample rate
#center_freq = 99.5e6 # station center frequency

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
    # FM Demodulation
    x = 0.5 * np.angle(x[0:-1] * np.conj(x[1:]))
    return x

def spectrum(samples, sample_rate):
    y = np.abs(np.fft.fft(samples, 2**14))**2 / (len(samples)*sample_rate)
    y = 10*np.log10(y)
    y = np.fft.fftshift(y)
    
    t = np.linspace(-sample_rate/2, sample_rate/2, len(y))
    
    
    plt.figure(figsize=(12, 4))
    plt.plot(t/1e3, y)
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
