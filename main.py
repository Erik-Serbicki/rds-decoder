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

def decode_rds(x):
    # Constants
    syndrome = [383, 14, 303, 663, 748]
    offset_pos = [0, 1, 2, 3, 2]
    offset_word = [252, 408, 360, 436, 848]

    # see Annex B, page 64 of the standard
    def calc_syndrome(x, mlen):
        reg = 0
        plen = 10
        for ii in range(mlen, 0, -1):
            reg = (reg << 1) | ((x >> (ii-1)) & 0x01)
            if (reg & (1 << plen)):
                reg = reg ^ 0x5B9
        for ii in range(plen, 0, -1):
            reg = reg << 1
            if (reg & (1 << plen)):
                reg = reg ^ 0x5B9
        return reg & ((1 << plen) - 1) # select the bottom plen bits of reg
    
    # Initialize all the working vars we'll need during the loop
    synced = False
    presync = False
    
    wrong_blocks_counter = 0
    blocks_counter = 0
    group_good_blocks_counter = 0
    
    reg = np.uint32(0) # was unsigned long in C++ (64 bits) but numpy doesn't support bitwise ops of uint64, I don't think it gets that high anyway
    lastseen_offset_counter = 0
    lastseen_offset = 0
    
    # the synchronization process is described in Annex C, page 66 of the standard */
    bytes_out = []
    for i in range(len(bits)):
        # in C++ reg doesn't get init so it will be random at first, for ours its 0s
        # It was also an unsigned long but never seemed to get anywhere near the max value
        # bits are either 0 or 1
        reg = np.bitwise_or(np.left_shift(reg, 1), bits[i]) # reg contains the last 26 rds bits. these are both bitwise ops
        if not synced:
            reg_syndrome = calc_syndrome(reg, 26)
            for j in range(5):
                if reg_syndrome == syndrome[j]:
                    if not presync:
                        lastseen_offset = j
                        lastseen_offset_counter = i
                        presync = True
                    else:
                        if offset_pos[lastseen_offset] >= offset_pos[j]:
                            block_distance = offset_pos[j] + 4 - offset_pos[lastseen_offset]
                        else:
                            block_distance = offset_pos[j] - offset_pos[lastseen_offset]
                        if (block_distance*26) != (i - lastseen_offset_counter):
                            presync = False
                        else:
                            print('Sync State Detected')
                            wrong_blocks_counter = 0
                            blocks_counter = 0
                            block_bit_counter = 0
                            block_number = (j + 1) % 4
                            group_assembly_started = False
                            synced = True
                break # syndrome found, no more cycles
    
        else: # SYNCED
            # wait until 26 bits enter the buffer */
            if block_bit_counter < 25:
                block_bit_counter += 1
            else:
                good_block = False
                dataword = (reg >> 10) & 0xffff
                block_calculated_crc = calc_syndrome(dataword, 16)
                checkword = reg & 0x3ff
                if block_number == 2: # manage special case of C or C' offset word
                    block_received_crc = checkword ^ offset_word[block_number]
                    if (block_received_crc == block_calculated_crc):
                        good_block = True
                    else:
                        block_received_crc = checkword ^ offset_word[4]
                        if (block_received_crc == block_calculated_crc):
                            good_block = True
                        else:
                            wrong_blocks_counter += 1
                            good_block = False
                else:
                    block_received_crc = checkword ^ offset_word[block_number] # bitwise xor
                    if block_received_crc == block_calculated_crc:
                        good_block = True
                    else:
                        wrong_blocks_counter += 1
                        good_block = False
    
                # Done checking CRC
                if block_number == 0 and good_block:
                    group_assembly_started = True
                    group_good_blocks_counter = 1
                    group = bytearray(8) # 8 bytes filled with 0s
                if group_assembly_started:
                    if not good_block:
                        group_assembly_started = False
                    else:
                        # raw data bytes, as received from RDS. 8 info bytes, followed by 4 RDS offset chars: ABCD/ABcD/EEEE (in US) which we leave out here
                        # RDS information words
                        # block_number is either 0,1,2,3 so this is how we fill out the 8 bytes
                        group[block_number*2] = (dataword >> 8) & 255
                        group[block_number*2+1] = dataword & 255
                        group_good_blocks_counter += 1
                        #print('group_good_blocks_counter:', group_good_blocks_counter)
                    if group_good_blocks_counter == 5:
                        #print(group)
                        bytes_out.append(group) # list of len-8 lists of bytes
                block_bit_counter = 0
                block_number = (block_number + 1) % 4
                blocks_counter += 1
                if blocks_counter == 50:
                    if wrong_blocks_counter > 35: # This many wrong blocks must mean we lost sync
                        print("Lost Sync (Got ", wrong_blocks_counter, " bad blocks on ", blocks_counter, " total)")
                        synced = False
                        presync = False
                    else:
                        print("Still Sync-ed (Got ", wrong_blocks_counter, " bad blocks on ", blocks_counter, " total)")
                    blocks_counter = 0
                    wrong_blocks_counter = 0

def parse_rds(bytes_out):
    # RDS parsing

    # Annex F of RBDS Standard Table F.1 (North America) and Table F.2 (Europe)
    #              Europe                   North America
    pty_table = [["Undefined",             "Undefined"],
                 ["News",                  "News"],
                 ["Current Affairs",       "Information"],
                 ["Information",           "Sports"],
                 ["Sport",                 "Talk"],
                 ["Education",             "Rock"],
                 ["Drama",                 "Classic Rock"],
                 ["Culture",               "Adult Hits"],
                 ["Science",               "Soft Rock"],
                 ["Varied",                "Top 40"],
                 ["Pop Music",             "Country"],
                 ["Rock Music",            "Oldies"],
                 ["Easy Listening",        "Soft"],
                 ["Light Classical",       "Nostalgia"],
                 ["Serious Classical",     "Jazz"],
                 ["Other Music",           "Classical"],
                 ["Weather",               "Rhythm & Blues"],
                 ["Finance",               "Soft Rhythm & Blues"],
                 ["Children’s Programmes", "Language"],
                 ["Social Affairs",        "Religious Music"],
                 ["Religion",              "Religious Talk"],
                 ["Phone-In",              "Personality"],
                 ["Travel",                "Public"],
                 ["Leisure",               "College"],
                 ["Jazz Music",            "Spanish Talk"],
                 ["Country Music",         "Spanish Music"],
                 ["National Music",        "Hip Hop"],
                 ["Oldies Music",          "Unassigned"],
                 ["Folk Music",            "Unassigned"],
                 ["Documentary",           "Weather"],
                 ["Alarm Test",            "Emergency Test"],
                 ["Alarm",                 "Emergency"]]
    pty_locale = 1 # set to 0 for Europe which will use first column instead
    
    # page 72, Annex D, table D.2 in the standard
    coverage_area_codes = ["Local",
                           "International",
                           "National",
                           "Supra-regional",
                           "Regional 1",
                           "Regional 2",
                           "Regional 3",
                           "Regional 4",
                           "Regional 5",
                           "Regional 6",
                           "Regional 7",
                           "Regional 8",
                           "Regional 9",
                           "Regional 10",
                           "Regional 11",
                           "Regional 12"]
    
    radiotext_AB_flag = 0
    radiotext = [' ']*65
    first_time = True
    for group in bytes_out:
        group_0 = group[1] | (group[0] << 8)
        group_1 = group[3] | (group[2] << 8)
        group_2 = group[5] | (group[4] << 8)
        group_3 = group[7] | (group[6] << 8)
    
        group_type = (group_1 >> 12) & 0xf # here is what each one means, e.g. RT is radiotext which is the only one we decode here: ["BASIC", "PIN/SL", "RT", "AID", "CT", "TDC", "IH", "RP", "TMC", "EWS", "___", "___", "___", "___", "EON", "___"]
        AB = (group_1 >> 11 ) & 0x1 # b if 1, a if 0
    
        #print("group_type:", group_type) # this is essentially message type, i only see type 0 and 2 in my recording
        #print("AB:", AB)
    
        program_identification = group_0     # "PI"
    
        program_type = (group_1 >> 5) & 0x1f # "PTY"
        pty = pty_table[program_type][pty_locale]
    
        pi_area_coverage = (program_identification >> 8) & 0xf
        coverage_area = coverage_area_codes[pi_area_coverage]
    
        pi_program_reference_number = program_identification & 0xff # just an int
    
        if first_time:
            print("PTY:", pty)
            print("program:", pi_program_reference_number)
            print("coverage_area:", coverage_area)
            first_time = False
    
        if group_type == 2:
            # when the A/B flag is toggled, flush your current radiotext
            if radiotext_AB_flag != ((group_1 >> 4) & 0x01):
                radiotext = [' ']*65
            radiotext_AB_flag = (group_1 >> 4) & 0x01
            text_segment_address_code = group_1 & 0x0f
            if AB:
                radiotext[text_segm
                radiotext[text_segment_address_code * 2 + 1] = chr(group_3        & 0xff)
            else:
                radiotext[text_segment_address_code *4     ] = chr((group_2 >> 8) & 0xff)
                radiotext[text_segment_address_code * 4 + 1] = chr(group_2        & 0xff)
                radiotext[text_segment_address_code * 4 + 2] = chr((group_3 >> 8) & 0xff)
                radiotext[text_segment_address_code * 4 + 3] = chr(group_3        & 0xff)
            print(''.join(radiotext))
        else:
            pass
            #print("unsupported group_type:", group_type)

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
    rds_text = decode_rds(bits)

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
