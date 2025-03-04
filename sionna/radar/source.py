import tensorflow as tf
import sionna
from sionna.constants import PI

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
    """
    FMCWSource generates FMCW (Frequency Modulated Continuous Wave) waveforms in baseband.
    
    This implementation incorporates properties similar to MATLAB's phased.FMCWWaveform:
      - sample_rate: Sample rate in Hz (default: 1e6)
      - sweep_time: Duration of each linear FM sweep in seconds (default: 1e-4)
      - sweep_bandwidth: FM sweep bandwidth in Hz (default: 1e5)
      - sweep_direction: FM sweep direction: 'Up' (default), 'Down', or 'Triangle'
      - sweep_interval: FM sweep interval: 'Positive' (default) or 'Symmetric'
      - output_format: 'Sweeps' (default) returns multiple sweeps or 'Samples' returns a concatenated vector
      - num_samples: Number of samples in output if output_format is 'Samples' (default: 100)
      - num_sweeps: Number of sweeps in output if output_format is 'Sweeps' (default: 1)

    Usage:
      # Generate multiple sweeps:
      fmcw = FMCWSource(sample_rate=1e6,
                        sweep_time=1e-4,
                        sweep_bandwidth=1e5,
                        sweep_direction='Up',
                        sweep_interval='Positive',
                        output_format='Sweeps',
                        num_sweeps=5)
      waveform = fmcw()  # waveform shape: [5, N]

      # Generate a single concatenated sample vector:
      fmcw = FMCWSource(sample_rate=1e6,
                        sweep_time=1e-4,
                        sweep_bandwidth=1e5,
                        sweep_direction='Up',
                        sweep_interval='Positive',
                        output_format='Samples',
                        num_samples=1000)
      waveform = fmcw()  # waveform is a 1D tensor with 1000 samples.
    """
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
        self.sample_rate = float(sample_rate)
        self.sweep_time = sweep_time          # can be scalar or vector (list/tuple)
        self.sweep_bandwidth = sweep_bandwidth  # can be scalar or vector
        self.sweep_direction = sweep_direction
        self.sweep_interval = sweep_interval
        self.output_format = output_format
        self.num_samples = num_samples
        self.num_sweeps = num_sweeps

        # Normalize sweep_time and sweep_bandwidth as lists for uniform processing.
        if not isinstance(self.sweep_time, (list, tuple)):
            self.sweep_time = [self.sweep_time]
        if not isinstance(self.sweep_bandwidth, (list, tuple)):
            self.sweep_bandwidth = [self.sweep_bandwidth]
        if len(self.sweep_time) != len(self.sweep_bandwidth):
            raise ValueError("sweep_time and sweep_bandwidth must have the same length if provided as vectors.")
        self.num_param = len(self.sweep_time)

    def _generate_sweep(self, sweep_time, sweep_bandwidth, direction):
        """
        Generate a single FMCW sweep with the specified sweep_time and sweep_bandwidth.
        
        For a 'Positive' sweep_interval:
          - 'Up': frequency sweeps from 0 to B with phase phi(t)= π * (B/T) * t^2.
          - 'Down': frequency sweeps from B to 0 with phase phi(t)= 2π*B*t - π*(B/T)*t^2.
          
        For a 'Symmetric' sweep_interval:
          - 'Up': frequency sweeps from -B/2 to B/2 with phase phi(t)= -π*B*t + π*(B/T)*t^2.
          - 'Down': frequency sweeps from B/2 to -B/2 with phase phi(t)= π*B*t - π*(B/T)*t^2.
        """
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
        
        # Generate the baseband complex FMCW signal: exp(j * phase)
        signal = tf.exp(tf.complex(tf.zeros_like(phase), phase))
        return signal

    def _generate_triangle_sweep(self, sweep_time, sweep_bandwidth, is_upsweep=True):
        """
        Generate a single sweep for a Triangle waveform.
        In Triangle mode, successive sweeps alternate between upsweep and downsweep.
        """
        direction = 'Up' if is_upsweep else 'Down'
        return self._generate_sweep(sweep_time, sweep_bandwidth, direction)

    def call(self, inputs=None):
        """
        Generate the FMCW waveform based on the configured properties.
        
        Returns
        -------
        If output_format is 'Sweeps': a tensor of shape [num_sweeps, N], where N is the number of samples per sweep.
        If output_format is 'Samples': a 1D tensor containing num_samples samples.
        """
        if self.output_format.lower() == 'sweeps':
            sweeps = []
            for i in range(self.num_sweeps):
                idx = i % self.num_param
                T = self.sweep_time[idx]
                B = self.sweep_bandwidth[idx]
                if self.sweep_direction.lower() == 'triangle':
                    is_upsweep = (i % 2 == 0)
                    sweep = self._generate_triangle_sweep(T, B, is_upsweep)
                else:
                    sweep = self._generate_sweep(T, B, self.sweep_direction)
                sweeps.append(sweep)
            waveform = tf.stack(sweeps, axis=0)
            return waveform
        elif self.output_format.lower() == 'samples':
            waveform_list = []
            total_samples = 0
            i = 0
            # Concatenate sweeps until reaching at least num_samples samples.
            while total_samples < self.num_samples:
                idx = i % self.num_param
                T = self.sweep_time[idx]
                B = self.sweep_bandwidth[idx]
                if self.sweep_direction.lower() == 'triangle':
                    is_upsweep = (i % 2 == 0)
                    sweep = self._generate_triangle_sweep(T, B, is_upsweep)
                else:
                    sweep = self._generate_sweep(T, B, self.sweep_direction)
                waveform_list.append(sweep)
                total_samples += tf.shape(sweep)[0]
                i += 1
            waveform_concat = tf.concat(waveform_list, axis=0)
            waveform = waveform_concat[:self.num_samples]
            return waveform
        else:
            raise ValueError("output_format must be either 'Sweeps' or 'Samples'.")
