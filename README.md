# RDS Decoder
Takes a simple FM broadcast IQ file, and decodes the RDS portion of the signal.
This is the radio station information, song info, etc.

### Usage
```bash
python main.py filename.iq -s sample_rate -f carrier_frequency -u units
```

Example with the included iq file on Linux:
```bash
python3 main.py samples.iq -s 250 -f 99.5 -u k
```
Specifies a sample rate of 250 kHz and a carrier frequency of 99.5 MHz. All FM radio stations are in the MHz range, so the units flag only specifies the scaling factor for the sample rate.

