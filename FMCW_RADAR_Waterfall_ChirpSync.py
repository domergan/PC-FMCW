# Imports
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from helpers import print_config, \
                    setup_phaser, \
                    setup_sdr, \
                    setup_tdd, \
                    end_program, \
                    update, \
                    RangeTimePlot,\
                    SpectrumPlot, \
                    MTIFilter

import adi


"""global params"""

sample_rate = 2e6
center_freq = 2.1e9
signal_freq = 100e3 # range bias (added to avoid DC noise, 1/f, 50Hz, ...)
rx_gain     = 30   # must be between -3 and 70
output_freq = 10e9
chirp_bw    = 300e6
ramp_time   = 500      # ramp time in us
plot_freq   = 200e3    # x-axis freq range to plot (eg max range)

USE_MTI     = False
MTI_MODE    = "ema"   # "ema" (background subtraction), "2pulse", or "3pulse"
MTI_ALPHA   = 0.08    # only for "ema": 0.01..0.2 typical

USE_CFAR = False

FAST_PLOTTING   = True    # turn on blitting
DECIMATE_RANGE  = 1       # plot every 2nd range bin
DECIMATE_SPEC   = 1       # plot every 2nd freq bin
SCROLL_LABELS   = False   # keep heatmap extent fixed (faster)
AUTO_SPEC_Y_EVERY = 60    # update spectrum y-lims every 60 frames (0=never)

rpi_ip = "ip:192.168.1.2"  
sdr_ip = "ip:192.168.2.1"  

my_sdr    = adi.ad9361(uri=sdr_ip)
my_phaser = adi.CN0566(uri=rpi_ip, sdr=my_sdr)

"""setups"""

setup_phaser(my_phaser, chirp_bw, output_freq, signal_freq, center_freq, ramp_time)
setup_sdr(my_sdr, sample_rate, center_freq, rx_gain)
tdd, num_chirps, sdr_pins = setup_tdd(sdr_ip, ramp_time)

"""determine sampling time"""

# From start of each ramp, how many "good" points do we want?
# For best freq linearity, stay away from the start of the ramps
ramp_time = int(my_phaser.freq_dev_time)
ramp_time_s = ramp_time / 1e6
begin_offset_time = 0.05 * ramp_time_s   # time in seconds
good_ramp_samples = int((ramp_time_s-begin_offset_time) * sample_rate)
start_offset_time = tdd.channel[0].on_ms/1e3 + begin_offset_time
start_offset_samples = int(start_offset_time * sample_rate)

print("actual freq dev time = {} µs".format(ramp_time))
print("begin offset time = {} s".format(begin_offset_time))
print("start offset samples = ", start_offset_samples)
print("good ramp samples = ", good_ramp_samples)

"""determine buffer sizes"""

# size the fft for the number of ramp data points
power=8
fft_size = int(2**power)
num_samples_frame = int(tdd.frame_length_ms/1000*sample_rate)
while num_samples_frame > fft_size:     
    power=power+1
    fft_size = int(2**power) 
    if power==18:
        break
    
# Pluto receive buffer size needs to be greater than total time for all chirps
total_time = tdd.frame_length_ms * num_chirps   # time in ms

buffer_time = 0
power=12
while total_time > buffer_time:     
    power=power+1
    buffer_size = int(2**power) 
    buffer_time = buffer_size/my_sdr.sample_rate*1000   # buffer time in ms
    if power==23:
        break     # max pluto buffer size is 2**23, but for tdd burst mode, set to 2**22
        
my_sdr.rx_buffer_size = buffer_size

"""print parameters"""

print_config(my_sdr, my_phaser, signal_freq, fft_size, good_ramp_samples, start_offset_samples)

""" Create a sinewave waveform for Pluto's transmitter"""

N = int(2**18)
fc = int(signal_freq)
ts = 1 / float(sample_rate)
t = np.arange(0, N * ts, ts)
i = np.cos(2 * np.pi * t * fc) * 2 ** 14
q = np.sin(2 * np.pi * t * fc) * 2 ** 14
iq = 1 * (i + 1j * q)

# transmit data from Pluto
my_sdr._ctx.set_timeout(30000)
my_sdr._rx_init_channels()
my_sdr.tx([iq, iq])

# Prime one frame
freq0, s_dbfs0 = update(my_sdr, my_phaser, num_chirps,
                        good_ramp_samples, start_offset_samples,
                        fft_size, num_samples_frame)
dt_frame = (tdd.frame_length_ms * num_chirps) / 1000.0

print("dt_frame =", dt_frame)

# Range–Time
vmin0, vmax0 = np.percentile(s_dbfs0[np.isfinite(s_dbfs0)], [5, 99])

rt_plot = RangeTimePlot(
    freq=freq0, chirp_bw=chirp_bw, ramp_time_s=ramp_time_s, dt=dt_frame,
    history=200, fmax=plot_freq,
    use_cfar=USE_CFAR,
    fast=FAST_PLOTTING, decimate=DECIMATE_RANGE, scroll_labels=SCROLL_LABELS,
    fixed_vmin=vmin0, fixed_vmax=vmax0, show_colorbar=True, offset_hz=signal_freq
)
rt_plot.push(freq0, s_dbfs0)

# Spectrum
spec_plot = SpectrumPlot(
    freq0, fmax=plot_freq, use_cfar=USE_CFAR,
    fast=FAST_PLOTTING, decimate=DECIMATE_SPEC, auto_ylim_every=AUTO_SPEC_Y_EVERY
)
spec_plot.update(freq0, s_dbfs0)

mti = MTIFilter(mode=MTI_MODE, alpha=MTI_ALPHA) if USE_MTI else None

plt.ion()
try:
    while True:
        freq, s_dbfs = update(my_sdr, my_phaser, num_chirps,
                              good_ramp_samples, start_offset_samples,
                              fft_size, num_samples_frame)
        
        if USE_MTI:
            s_dbfs = mti.process_db(s_dbfs)
    
        rt_plot.push(freq, s_dbfs)
        spec_plot.update(freq, s_dbfs)
        plt.pause(0.01)  
except KeyboardInterrupt:
    end_program(my_sdr, tdd, sdr_pins)



    