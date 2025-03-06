import tensorflow as tf
import numpy as np
import sionna
from sionna.constants import PI
from matplotlib import pyplot as plt
from scipy.signal import spectrogram


class RadarSource(tf.keras.layers.Layer):
    """
    RadarSource(dtype=tf.float32, seed=None, **kwargs)

    Base layer for generating radar signals.
    This class should be extended to implement specific radar signal types.

    Parameters
    ----------
    dtype : tf.DType
        Defines the output data type of the layer.
        Default: tf.float32.
    seed : int or None
        Seed for the random number generator.
        If None, the default Sionna RNG (sionna.config.tf_rng) is used.

    Input
    -----
    inputs : tensor or list/array
        Parameters that define the shape or characteristics of the generated signal.

    Output
    ------
    Tensor with the specified shape and data type, containing the generated signal.
    """
    def __init__(self, dtype=tf.float32, seed=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self._seed = seed
        if self._seed is not None:
            self._rng = tf.random.Generator.from_seed(self._seed)
        else:
            self._rng = sionna.config.tf_rng

    def call(self, inputs):
        raise NotImplementedError("The 'call' method is not implemented. "
                                  "Please use a derived class to generate specific signals.")

class FMCWSource(RadarSource):
    def __init__(self,
                 sample_rate=1e6,
                 sweep_time=1e-4,
                 sweep_bandwidth=1e5,
                 sweep_direction='Up',
                 sweep_interval='Positive',
                 output_format='Sweeps',
                 num_samples=100,
                 num_sweeps=1,
                 **kwargs):
        # For baseband signals, default dtype is complex64.
        if "dtype" not in kwargs:
            kwargs["dtype"] = tf.complex64
        super().__init__(**kwargs)

        # Validate inputs
        if sweep_direction.lower() not in ['up', 'down', 'triangle']:
            raise ValueError("sweep_direction must be 'Up', 'Down', or 'Triangle'.")
        if sweep_interval.lower() not in ['positive', 'symmetric']:
            raise ValueError("sweep_interval must be 'Positive' or 'Symmetric'.")
        if output_format.lower() not in ['sweeps', 'samples']:
            raise ValueError("output_format must be 'Sweeps' or 'Samples'.")

        self.sample_rate = float(sample_rate)
        self.sweep_time = sweep_time if isinstance(sweep_time, (list, tuple)) else [sweep_time]
        self.sweep_bandwidth = sweep_bandwidth if isinstance(sweep_bandwidth, (list, tuple)) else [sweep_bandwidth]
        self.sweep_direction = sweep_direction
        self.sweep_interval = sweep_interval
        self.output_format = output_format
        self.num_samples = num_samples
        self.num_sweeps = num_sweeps

        # Validate sweep_time and sweep_bandwidth
        if len(self.sweep_time) != len(self.sweep_bandwidth):
            raise ValueError("sweep_time and sweep_bandwidth must have the same length if provided as vectors.")
        for t in self.sweep_time:
            if not (self.sample_rate * t).is_integer():
                raise ValueError("(sample_rate * sweep_time) must result in an integer for all sweep times.")
        # Define num_param as the length of sweep_time or sweep_bandwidth
        self.num_param = len(self.sweep_time)

    def _generate_sweep(self, sweep_time, sweep_bandwidth, direction):
        """Generate a single FMCW sweep."""
        N = int(round(self.sample_rate * sweep_time))
        t = tf.linspace(0.0, sweep_time, N)
        T = sweep_time
        B = sweep_bandwidth
        
        if self.sweep_interval.lower() == 'positive':
            if direction.lower() == 'up':
                phase = PI * (B / T) * tf.square(t)
            elif direction.lower() == 'down':
                phase = 2 * PI * B * t - PI * (B / T) * tf.square(t)
            else:
                raise ValueError("For 'Positive' interval, direction must be 'Up' or 'Down'.")
        elif self.sweep_interval.lower() == 'symmetric':
            if direction.lower() == 'up':
                phase = -PI * B * t + PI * (B / T) * tf.square(t)
            elif direction.lower() == 'down':
                phase = PI * B * t - PI * (B / T) * tf.square(t)
            else:
                raise ValueError("For 'Symmetric' interval, direction must be 'Up' or 'Down'.")
        else:
            raise ValueError("sweep_interval must be 'Positive' or 'Symmetric'.")
        
        signal = tf.exp(tf.complex(tf.zeros_like(phase), phase))
        return signal

    def _generate_triangle_sweep(self, sweep_time, sweep_bandwidth, is_upsweep=True):
        """Generate a single sweep for a Triangle waveform."""
        direction = 'Up' if is_upsweep else 'Down'
        return self._generate_sweep(sweep_time, sweep_bandwidth, direction)

    def call(self, inputs=None):
        """Generate the FMCW waveform based on the configured properties."""
        if self.output_format.lower() == 'sweeps':
            sweeps = []
            for i in range(self.num_sweeps):
                idx = i % self.num_param
                T = self.sweep_time[idx]
                B = self.sweep_bandwidth[idx]
                
                # Ajustar o tempo com base na direção da varredura
                if self.sweep_direction.lower() == 'triangle':
                    # Para 'Triangle', o sweep_time é metade do período completo
                    T_sweep = T  # T_sweep é o tempo de meia varredura
                    is_upsweep = (i % 2 == 0)
                    sweep = self._generate_triangle_sweep(T_sweep, B, is_upsweep)
                else:
                    # Para 'Up' ou 'Down', o sweep_time é o tempo de uma varredura completa
                    T_sweep = T
                    sweep = self._generate_sweep(T_sweep, B, self.sweep_direction)
                
                sweeps.append(sweep)
            waveform = tf.stack(sweeps, axis=0)
            return waveform
        elif self.output_format.lower() == 'samples':
            waveform_list = []
            total_samples = 0
            i = 0
            while total_samples < self.num_samples:
                idx = i % self.num_param
                T = self.sweep_time[idx]
                B = self.sweep_bandwidth[idx]
                
                # Ajustar o tempo com base na direção da varredura
                if self.sweep_direction.lower() == 'triangle':
                    # Para 'Triangle', o sweep_time é metade do período completo
                    T_sweep = T
                    is_upsweep = (i % 2 == 0)
                    sweep = self._generate_triangle_sweep(T_sweep, B, is_upsweep)
                else:
                    # Para 'Up' ou 'Down', o sweep_time é o tempo de uma varredura completa
                    T_sweep = T
                    sweep = self._generate_sweep(T_sweep, B, self.sweep_direction)
                
                waveform_list.append(sweep)
                total_samples += tf.shape(sweep)[0]
                i += 1
            waveform_concat = tf.concat(waveform_list, axis=0)
            waveform = waveform_concat[:self.num_samples]
            return waveform
        else:
            raise ValueError("output_format must be either 'Sweeps' or 'Samples'.")  

if __name__ == '__main__':
    # Exemplo equivalente ao MATLAB:
    # waveform = phased.FMCWWaveform('SweepBandwidth', 100.0e3, ...
    #                               'OutputFormat', 'Sweeps', 'NumSweeps', 2);
    fmcw = FMCWSource(
        sample_rate=1e6,          # Taxa de amostragem de 1 MHz
        sweep_time=1e-4,          # Tempo de varredura de 100 μs
        sweep_bandwidth=100.0e3,  # Largura de banda de varredura de 100 kHz
        sweep_direction='Up',     # Direção da varredura (para cima)
        sweep_interval='Positive', # Intervalo de varredura positivo [0, B]
        output_format='Sweeps',   # Formato de saída: múltiplas varreduras
        num_sweeps=2              # Número de varreduras
    )
    waveform = fmcw(None)  # Passar None como argumento para o método call

    # Plot da parte real do sinal no domínio do tempo
    t = tf.linspace(0.0, fmcw.num_sweeps * fmcw.sweep_time[0], waveform.shape[1]).numpy()  # Tempo para 2 varreduras
    plt.figure(figsize=(10, 4))
    plt.plot(t, tf.math.real(waveform[0]).numpy(), label='Sweep 1')
    plt.plot(t, tf.math.real(waveform[1]).numpy(), label='Sweep 2')
    plt.title("FMCW Signal - Real Part (Time Domain)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()


# ------------------------
    # Now, add the spectrogram plot.
    # We'll use the first sweep for the spectrogram.
    sweep_signal = waveform[0].numpy()  # Convert first sweep to NumPy array

    # For a signal with 100 samples, choose a window size (nperseg) smaller than 100.
    nperseg = 64
    noverlap = 32  # Ensure noverlap < nperseg

    # Compute the spectrogram; we use the squared magnitude to represent power.
    f_spec, t_spec, Sxx = spectrogram(
        x=np.real(sweep_signal),  # Use the real part of the complex signal
        fs=fmcw.sample_rate, 
        nperseg=nperseg, 
        noverlap=noverlap, 
        scaling='density'
    )
    # Convert the power spectral density to dB/Hz
    Sxx_dB = 10 * np.log10(Sxx + 1e-12)  # small offset to avoid log(0)

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t_spec, f_spec/1e3, Sxx_dB, shading='gouraud', cmap='viridis')
    plt.title("Spectrogram of FMCW Signal (Intensity in dB/Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (KHz)")
    plt.ylim(0, 200)  # Limit the frequency axis from 0 to 100 kHz
    cbar = plt.colorbar()
    cbar.set_label("Intensity [dB/Hz]")
    plt.tight_layout()
    plt.show()