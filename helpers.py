# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 12:38:22 2025

@author: Bertrand
"""

import matplotlib.pyplot as plt
import scipy as sc
import numpy as np
import adi

from mmwave.dsp.cfar import os

from phase_code import design_group_delay_allpass, make_tx_buffer


CFAR_METHOD  = 'false_alarm'   # 'average' | 'greatest' | 'smallest' | 'false_alarm'
CFAR_BIAS    = 0.065         # additive bias in *linear amplitude* units
CFAR_FA_RATE = 0.2         # used only when CFAR_METHOD == 'false_alarm'
CFAR_GUARD = 10
CFAR_NOISE = 12


def cfar(X_k, num_guard_cells, num_ref_cells, bias=1, cfar_method='average', fa_rate=0.2):
    N = X_k.size
    cfar_values = np.ma.masked_all(X_k.shape)
    for center_index in range(num_guard_cells + num_ref_cells, N - (num_guard_cells + num_ref_cells)):
        min_index = center_index - (num_guard_cells + num_ref_cells)
        min_guard = center_index - num_guard_cells 
        max_index = center_index + (num_guard_cells + num_ref_cells) + 1
        max_guard = center_index + num_guard_cells + 1

        lower_nearby = X_k[min_index:min_guard]
        upper_nearby = X_k[max_guard:max_index]

        lower_mean = np.mean(lower_nearby)
        upper_mean = np.mean(upper_nearby)

        if (cfar_method == 'average'):
            mean = np.mean(np.concatenate((lower_nearby, upper_nearby)))
            output = mean + bias
        elif (cfar_method == 'greatest'):
            mean = max(lower_mean, upper_mean)
            output = mean + bias
        elif (cfar_method == 'smallest'):
            mean = min(lower_mean, upper_mean)
            output = mean + bias
        elif (cfar_method == 'false_alarm'):
            refs = np.concatenate((lower_nearby, upper_nearby))
            noise_variance = np.sum(refs**2 / refs.size)
            output = (noise_variance * -2 * np.log(fa_rate))**0.5
        else:
            raise Exception('No CFAR method received')

        cfar_values[center_index] = output

    cfar_values[np.where(cfar_values == np.ma.masked)] = np.min(cfar_values)

    targets_only = np.ma.masked_array(np.copy(X_k))
    targets_only[np.where(abs(X_k) > abs(cfar_values))] = np.ma.masked

    if (cfar_method == 'false_alarm'):
        return cfar_values, targets_only, noise_variance
    else:
        return cfar_values, targets_only


def print_config(sdr, phaser, signal_freq, fft_size, good_ramp_samples, start_offset_samples):
    c = sc.constants.c  
    
    center_freq = sdr.rx_lo
    sample_rate = sdr.sample_rate
    rx_gain = sdr.rx_hardwaregain_chan0
    output_freq = phaser.frequency * 4
    bw_hz = phaser.freq_dev_range * 4
    rt_s = phaser.freq_dev_time
    
    range_res_m = c / (2 * bw_hz)
    wavelength_m = c / float(output_freq)
    
    print("=== CN0566/Pluto Configuration ===")
    print(f"Sample rate:            {sample_rate/1e6:,.3f} MHz")
    print(f"Center frequency (LO):  {center_freq/1e9:,.3f} GHz")
    print(f"IF/baseband tone:       {signal_freq/1e3:,.1f} kHz")
    print(f"RX gain:                {rx_gain} dB")
    print(f"Output frequency:       {output_freq/1e9:,.3f} GHz")
    print(f"Element spacing:        {getattr(phaser, 'element_spacing', float('nan')):.4f} m")
    print(f"Chirp bandwidth:        {bw_hz/1e6:,.1f} MHz")
    print(f"Ramp time:              {rt_s:,.3f} ms")
    print(f"Wavelength (@output):   {wavelength_m*1e3:,.3f} mm")
    print(f"FFT size:               {fft_size}")
    print(f"RX buffer size:         {sdr.rx_buffer_size} samples")
    print(f"Good ramp samples:      {good_ramp_samples}")
    print(f"Start offset samples:   {start_offset_samples}")
    print("\n— Derived —")
    print(f"Range resolution ΔR = c/(2·B): {range_res_m:.4f} m ({range_res_m*100:.2f} cm)")


def setup_phaser(phaser, default_chirp_bw, output_freq, signal_freq, center_freq, ramp_time):
    phaser.configure(device_mode="rx")
    phaser.element_spacing = 0.014

    phaser.load_gain_cal()
    phaser.load_phase_cal()
    
    for i in range(0, 8):
        phaser.set_chan_phase(i, 0)
    
    gain_list = [8, 34, 84, 127, 127, 84, 34, 8]  # Blackman taper
    for i in range(0, len(gain_list)):
        phaser.set_chan_gain(i, gain_list[i], apply_cal=True)

    # Setup Raspberry Pi GPIO states
    phaser._gpios.gpio_tx_sw = 0  # 0 = TX_OUT_2, 1 = TX_OUT_1
    phaser._gpios.gpio_vctrl_1 = 1 # 1=Use onboard PLL/LO source  (0=disable PLL and VCO, and set switch to use external LO input)
    phaser._gpios.gpio_vctrl_2 = 1 # 1=Send LO to transmit circuitry  (0=disable Tx path, and send LO to LO_OUT)

    # Configure the ADF4159 Ramping PLL
    vco_freq = int(output_freq + signal_freq + center_freq)
    BW = default_chirp_bw
    num_steps = int(ramp_time)    # in general it works best if there is 1 step per us
    
    phaser.frequency = int(vco_freq / 4)
    phaser.freq_dev_range = int(BW / 4)      # total freq deviation of the complete freq ramp in Hz
    phaser.freq_dev_step = int((BW / 4) / num_steps)  # This is fDEV, in Hz.  Can be positive or negative
    phaser.freq_dev_time = int(ramp_time)  # total time (in us) of the complete frequency ramp
    
    print("requested freq dev time = ", ramp_time)
    
    phaser.delay_word = 4095  # 12 bit delay word.  4095*PFD = 40.95 us.  For sawtooth ramps, this is also the length of the Ramp_complete signal
    phaser.delay_clk = "PFD"  # can be 'PFD' or 'PFD*CLK1'
    phaser.delay_start_en = 0  # delay start
    phaser.ramp_delay_en = 0  # delay between ramps.
    phaser.trig_delay_en = 0  # triangle delay
    phaser.ramp_mode = "single_sawtooth_burst"  # ramp_mode can be:  "disabled", "continuous_sawtooth", "continuous_triangular", "single_sawtooth_burst", "single_ramp_burst"
    phaser.sing_ful_tri = 0  # full triangle enable/disable -- this is used with the single_ramp_burst mode
    phaser.tx_trig_en = 1  # start a ramp with TXdata
    phaser.enable = 0  # 0 = PLL enable.  Write this last to update all the registers
    
def setup_sdr(sdr, sample_rate, center_freq, rx_gain):
    
    # Configure SDR Rx
    sdr.sample_rate = int(sample_rate)
    sample_rate = int(sdr.sample_rate)
    sdr.rx_lo = int(center_freq)
    sdr.rx_enabled_channels = [0, 1]  # enable Rx1 and Rx2
    sdr.gain_control_mode_chan0 = "manual"  # manual or slow_attack
    sdr.gain_control_mode_chan1 = "manual"  # manual or slow_attack
    sdr.rx_hardwaregain_chan0 = int(rx_gain)  # must be between -3 and 70
    sdr.rx_hardwaregain_chan1 = int(rx_gain)  # must be between -3 and 70
    
    # Configure SDR Tx
    sdr.tx_lo = int(center_freq)
    sdr.tx_enabled_channels = [0, 1]
    sdr.tx_cyclic_buffer = True  # must set cyclic buffer to true for the tdd burst mode.
    sdr.tx_hardwaregain_chan0 = -88  # must be between 0 and -88
    sdr.tx_hardwaregain_chan1 = -0  # must be between 0 and -88
    
def setup_tdd(sdr_ip, ramp_time):
    
    sdr_pins = adi.one_bit_adc_dac(sdr_ip)
    
    # If set to True, this enables external capture triggering using the L24N GPIO on the Pluto.  
    # When set to false, an internal trigger pulse will be generated every second
    sdr_pins.gpio_tdd_ext_sync = True 
    
    tdd = adi.tddn(sdr_ip)
    sdr_pins.gpio_phaser_enable = True
    tdd.enable = False         # disable TDD to configure the registers
    tdd.sync_external = True
    
    # Initial delay before the first frame (ms)
    tdd.startup_delay_ms = 0
    
    # in ms (0.5 + 1 = 1.5ms)
    PRI_ms = ramp_time/1e3 + 1.0 
    
    # each chirp is spaced this far apart
    tdd.frame_length_ms = PRI_ms    
    
    # Amount of frames to produce, where 0 means repeat indefinitely
    tdd.burst_count = 1 # one chirp per burst
    
    """
    The Generic TDD Engine was integrated to output a logic signal on the L10P pin, which connects to the input of the ADF4159,
    when receiving an external synchronization signal on the L12N pin from the Raspberry Pi. Two additional TDD channels are used
    to synchronize the TX/RX DMA transfer start:
        - TDD CH1 is connected to the RX DMA, triggering the synchronization flag;
        - TDD CH2 is connected to the TX unpacker's reset, backpressuring the TX DMA until deasserted.
    """
    
    # Channel 0 controls TXDATA so the chirp generation by the PLL
    # TXDATA_1V8, Pluto L10P pin
    tdd.channel[0].enable   = True
    tdd.channel[0].polarity = False
    tdd.channel[0].on_raw   = 0
    tdd.channel[0].off_raw  = 10
    
    # Channel 1 controls the start timing for pluto RX buffer
    # RX DMA transfer start sync
    tdd.channel[1].enable   = True
    tdd.channel[1].polarity = False 
    tdd.channel[1].on_raw   = 0
    tdd.channel[1].off_raw  = 10
    
    # Channel 2 controls the start timing for pluto TX buffer
    # TX DMA SYNC
    tdd.channel[2].enable   = True
    tdd.channel[2].polarity = False
    tdd.channel[2].on_raw   = 0
    tdd.channel[2].off_raw  = 10
    
    tdd.enable = True
    
    return tdd, sdr_pins
    
def end_program(sdr, tdd, sdr_pins):
    """ Gracefully shutsdown the program and Pluto
    """
    
    sdr.tx_destroy_buffer()
    
    print("Program finished and Pluto Tx Buffer Cleared")
    
    # disable TDD and revert to non-TDD (standard) mode
    tdd.enable = False
    sdr_pins.gpio_phaser_enable = False
    tdd.channel[1].polarity = not(sdr_pins.gpio_phaser_enable)
    tdd.channel[2].polarity = sdr_pins.gpio_phaser_enable
    tdd.enable = True
    tdd.enable = False
    

def update(sdr, phaser, good_ramp_samples, start_offset_samples, fft_size, fs, K):
    """ Updates the FFT in the window
	"""
    # toggles low->high->low from the raspberry pi
    # this triggers a ramp and buffer start (Pluto L12N pin)
    phaser._gpios.gpio_burst = 0
    phaser._gpios.gpio_burst = 1
    phaser._gpios.gpio_burst = 0
    
    # sample from SDR
    data = sdr.rx()
    chan1 = data[0]
    chan2 = data[1]
    sum_data = chan1+chan2

    # select just the linear portion of the last chirp
    rx_bursts   = np.zeros(good_ramp_samples, dtype=complex)
    start_index = start_offset_samples
    stop_index  = start_index + good_ramp_samples
    rx_bursts   = sum_data[start_index:stop_index]
    burst_data  = np.ones(fft_size, dtype=complex)*1e-10
    win_funct   = np.blackman(len(rx_bursts))
    
    burst_data[start_offset_samples:(start_offset_samples+good_ramp_samples)] = rx_bursts*win_funct
    
    # apply filter on good samples
    H, f, phi = design_group_delay_allpass(len(burst_data), fs, K, f_max=fs/2)
    X = np.fft.fft(burst_data)
    Y = H * X
    burst_data = np.fft.ifft(Y) 
    
    
    # burst_data should be aligned be now  
    
    # phases = [np.pi, 0, np.pi, 0, np.pi]
    
    # tx_ref, _, _ = make_tx_buffer(
    #     phases=phases,   # your phase code per chip
    #     fc=0.0,          # IMPORTANT: 0 so we only get the code envelope
    #     fs=fs,
    #     num_samples=good_ramp_samples,
    #     start_phase=0.0
    # )

    # # ... after: burst_data = np.fft.ifft(Y)
    # start = start_offset_samples
    # stop  = start + good_ramp_samples
    
    # # decoding
    # burst_data[start:stop] *= tx_ref


    # magnitude of fft
    sp = np.absolute(np.fft.fft(burst_data))
    sp = np.fft.fftshift(sp)
    s_mag = np.abs(sp) / np.sum(win_funct)
    s_mag = np.maximum(s_mag, 10 ** (-15))
    s_dbfs = 20 * np.log10(s_mag / (2 ** 11))
    
    # frequency range from fs
    freq = np.linspace(-fs / 2, fs / 2, int(fft_size))
    
    # return burst_data too for debugging purposes
    return freq, s_dbfs, burst_data


class RangeTimePlot:
    def __init__(self, freq, chirp_bw, ramp_time_s, dt, history=200, fmax=None,
                 use_cfar=False,
                 fast=True, decimate=1, scroll_labels=False,
                 fixed_vmin=None, fixed_vmax=None, show_colorbar=True, offset_hz=0.0):
        """
        fast: blit-only updates (huge speedup)
        decimate: plot every Nth range bin to lighten draw
        scroll_labels: False keeps a fixed x-axis (-T..0) so blit background stays valid
        fixed_vmin/fixed_vmax: lock color scale (avoid per-frame clim updates)
        """
        
        self.fast = bool(fast)
        self.scroll_labels = bool(scroll_labels)
        self.dt = float(dt)
        self.history = int(history)
        self.c = 299_792_458.0
        self.slope = chirp_bw / float(ramp_time_s)
        self.last_time = 0.0
        
        self.offset_hz = float(offset_hz)

        # replace your frequency selection to use the offset-corrected grid
        f_eff = freq - self.offset_hz
        mask = f_eff >= 0
        if fmax is not None:
            mask &= (f_eff <= fmax)
        
        self.f_sel = f_eff[mask]              # keep your existing decimation if you have it
        self.ranges_m = 299_792_458.0 * self.f_sel / (2.0 * (chirp_bw / ramp_time_s))

        # freq selection (positive beat freq, up to fmax)
        mask = freq >= 0
        if fmax is not None:
            mask &= (freq <= fmax)
        fsel = freq[mask]
        if fsel.size == 0:
            raise ValueError("RangeTimePlot: empty frequency selection (check fmax).")

        # decimate range axis if asked
        fsel = fsel[::max(1, int(decimate))]
        self.f_sel = fsel
        self.ranges_m = self.c * fsel / (2.0 * self.slope)

        # rolling buffers
        self.buf = np.full((fsel.size, self.history), np.nan, dtype=np.float32)
        self.hit_buf = np.zeros_like(self.buf, dtype=bool)

        # figure/axes
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Range–Time (FMCW)")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Range (m)")

        T = self.history * self.dt
        extent = [-T, 0.0, self.ranges_m[0], self.ranges_m[-1]]

        self.im = self.ax.imshow(
            self.buf, origin="lower", aspect="auto", extent=extent,
            interpolation="nearest",  # faster
            vmin=fixed_vmin, vmax=fixed_vmax, animated=True
        )
        # Optional colorbar (static)
        if show_colorbar:
            self.fig.colorbar(self.im, ax=self.ax, label="dBFS")

        # CFAR overlay: use imshow mask for speed (animated too)
        self.hit_im = self.ax.imshow(
            self.hit_buf.astype(float), origin="lower", aspect="auto", extent=extent,
            interpolation="nearest", cmap="Reds", vmin=0, vmax=1, alpha=0.35, animated=True
        )

        # CFAR config
        self.use_cfar = bool(use_cfar)

        # Prepare blitting
        self._background = None
        self.fig.canvas.draw()  # first full draw
        self._capture_background()
        self.fig.canvas.mpl_connect("resize_event", self._on_resize)

    def push(self, freq, s_dbfs):
        # interp onto our fixed range grid if needed
        freq_eff = freq - self.offset_hz
        if freq_eff.shape != self.f_sel.shape or not np.allclose(freq_eff, self.f_sel):
            col = np.interp(self.f_sel, freq_eff, s_dbfs)
        else:
            col = s_dbfs
        # (leave the rest of push() unchanged)


        # roll buffer
        self.buf[:, :-1] = self.buf[:, 1:]
        self.buf[:, -1] = col

        # CFAR (log domain)
        if self.use_cfar:
            col_lin = np.power(10.0, col / 20.0)
            thr, _ = cfar(col_lin,
                          num_guard_cells=CFAR_GUARD,
                          num_ref_cells=CFAR_NOISE,
                          bias=CFAR_BIAS,
                          cfar_method=CFAR_METHOD,
                          fa_rate=CFAR_FA_RATE)
            hits = np.abs(col_lin) > np.abs(thr)
        else:
            hits = np.zeros_like(col, dtype=bool)
        self.hit_buf[:, :-1] = self.hit_buf[:, 1:]
        self.hit_buf[:, -1] = hits

        # update images
        self.im.set_data(self.buf)
        self.hit_im.set_data(self.hit_buf.astype(float))

        # keep labels static for blit speed OR scroll if you want (slower)
        if self.scroll_labels:
            self.last_time += self.dt
            T = self.history * self.dt
            extent = [self.last_time - T, self.last_time, self.ranges_m[0], self.ranges_m[-1]]
            self.im.set_extent(extent)
            self.hit_im.set_extent(extent)

        # blit-only update (fast path)
        if self.fast and self._background is not None:
            canvas = self.fig.canvas
            canvas.restore_region(self._background)
            self.ax.draw_artist(self.im)
            self.ax.draw_artist(self.hit_im)
            canvas.blit(self.ax.bbox)
            canvas.flush_events()
        else:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    # easy helpers
    def _capture_background(self):
        # store the static background (axes frame, ticks, etc.)
        self._background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def _on_resize(self, _evt):
        # on resize, we must redraw once and recapture background
        self.fig.canvas.draw()
        self._capture_background()


class SpectrumPlot:
    def __init__(self, freq, fmax=None, use_cfar=False,
                 fast=True, decimate=1, title="s_dbfs vs Frequency",
                 auto_ylim_every=0):
        """
        fast: blit updates
        decimate: plot every Nth point for speed
        auto_ylim_every: recompute y-limits every N frames (0=never)
        """
        self.fast = bool(fast)
        self.use_cfar = bool(use_cfar)
        self.auto_ylim_every = int(auto_ylim_every)
        self.frame_i = 0

        mask = freq >= 0
        if fmax is not None:
            mask &= (freq <= fmax)
        fsel = freq[mask][::max(1, int(decimate))]
        if fsel.size == 0:
            raise ValueError("SpectrumPlot: empty frequency selection (check fmax).")
        self.f_sel = fsel

        # units
        fmax_val = float(fsel.max())
        if fmax_val >= 1e6:
            self.x_scale, self.x_label = 1e6, "Beat frequency (MHz)"
        elif fmax_val >= 1e3:
            self.x_scale, self.x_label = 1e3, "Beat frequency (kHz)"
        else:
            self.x_scale, self.x_label = 1.0, "Beat frequency (Hz)"
        self.x = fsel / self.x_scale

        self.fig, self.ax = plt.subplots()
        self.ax.set_title(title)
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel("Amplitude (dBFS)")

        (self.line,) = self.ax.plot(self.x, np.full_like(self.x, np.nan), lw=1, animated=True)

        # Use a Line2D for CFAR markers (faster than scatter)
        self.hit_line, = self.ax.plot([], [], "o", ms=3, mfc="none", mec="r",
                                      linestyle="none", animated=True, label="CFAR")
        if self.use_cfar:
            self.ax.legend(loc="best")

        # Prepare blitting
        self._background = None
        self.fig.canvas.draw()
        self._capture_background()
        self.fig.canvas.mpl_connect("resize_event", self._on_resize)

        # Fix x limits (static)
        self.ax.set_xlim(self.x[0], self.x[-1])

    def update(self, freq, s_dbfs):
        # interp onto fixed x-grid if needed
        if freq.shape != self.f_sel.shape or not np.allclose(freq, self.f_sel):
            y = np.interp(self.f_sel, freq, s_dbfs)
        else:
            y = s_dbfs

        self.line.set_ydata(y)

        if self.use_cfar:
            y_lin = np.power(10.0, y / 20.0)
            thr, _ = cfar(y_lin,
                          num_guard_cells=CFAR_GUARD,
                          num_ref_cells=CFAR_NOISE,
                          bias=CFAR_BIAS,
                          cfar_method=CFAR_METHOD,
                          fa_rate=CFAR_FA_RATE)
            det = np.abs(y_lin) > np.abs(thr)
            self.hit_line.set_data(self.x[det], y[det])  # plot dots at dB y-values

        # recompute y-lims 
        if self.auto_ylim_every and (self.frame_i % self.auto_ylim_every == 0):
            finite = np.isfinite(y)
            if finite.any():
                lo = np.nanpercentile(y[finite], 5)
                hi = np.nanpercentile(y[finite], 99)
                pad = 0.1 * (hi - lo + 1e-6)
                self.ax.set_ylim(lo - pad, hi + pad)

        # blit-only update
        if self.fast and self._background is not None:
            canvas = self.fig.canvas
            canvas.restore_region(self._background)
            self.ax.draw_artist(self.line)
            if self.use_cfar:
                self.ax.draw_artist(self.hit_line)
            canvas.blit(self.ax.bbox)
            canvas.flush_events()
        else:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        self.frame_i += 1

    def _capture_background(self):
        self._background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def _on_resize(self, _evt):
        self.fig.canvas.draw()
        self._capture_background()

class MTIFilter:
    """
    Slow-time MTI on spectrum frames.
    Operates in linear power, returns dB.
    Modes:
      - 'ema'   : y = x - EMA(x)         (good general clutter removal)
      - '2pulse': y[n] = x[n] - x[n-1]   (classic 2-pulse canceller)
      - '3pulse': y[n] = x[n] - 2x[n-1] + x[n-2]  (steeper HPF)
    """
    def __init__(self, mode="ema", alpha=0.05):
        self.mode = mode
        self.alpha = float(alpha)
        self.bg = None
        self.prev = None
        self.prev2 = None
        self._eps = 1e-12  # power floor to avoid log(0)

    def process_db(self, x_db):
        x_lin = np.power(10.0, x_db / 10.0)  # dBFS -> linear power

        if self.mode == "2pulse":
            if self.prev is None:
                y_lin = np.zeros_like(x_lin)
            else:
                y_lin = x_lin - self.prev
            self.prev = x_lin

        elif self.mode == "3pulse":
            if self.prev is None:
                self.prev = x_lin
                self.prev2 = np.zeros_like(x_lin)
                y_lin = np.zeros_like(x_lin)
            else:
                if self.prev2 is None:
                    self.prev2 = self.prev
                y_lin = x_lin - 2.0 * self.prev + self.prev2
                self.prev2, self.prev = self.prev, x_lin

        else:  # 'ema' background subtraction
            if self.bg is None:
                self.bg = x_lin.copy()
            else:
                self.bg = (1.0 - self.alpha) * self.bg + self.alpha * x_lin
            y_lin = x_lin - self.bg

        y_lin = np.clip(y_lin, self._eps, None)
        return 10.0 * np.log10(y_lin)  # back to dB

