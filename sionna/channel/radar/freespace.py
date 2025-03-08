#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Free-space radar channel model implementation."""

import tensorflow as tf
import math
from sionna.channel.channel_model import ChannelModel
from sionna.constants import PI, SPEED_OF_LIGHT

class RadarFreespaceChannel(ChannelModel):
    """
    Free-space channel model for radar applications.

    This model computes the channel coefficient for a single TX-RX pair using the free-space
    propagation model. The coefficient is calculated based on the distance between the target
    and the transmitter (assumed to be at the origin) for the monostatic case or using both the
    transmitter-to-target and target-to-receiver distances for the bistatic case.

    Attributes:
        radar_target: An instance of a radar target (e.g., CarTarget) providing target properties.
        monostatic (bool): True if a monostatic configuration is assumed (default). In monostatic mode,
                           the receiver is assumed to be co-located with the transmitter.
        rx_position (tf.Tensor or None): For bistatic configuration, the receiver position as a tensor
                           with shape [batch_size, 3]. Not used for monostatic mode.
        carrier_frequency (float): The radar carrier frequency in Hz.
        wavelength (float): The wavelength computed from the carrier frequency.
    """
    def __init__(self, radar_target, monostatic=True, rx_position=None, carrier_frequency=76.5e9):
        """
        Initialize the RadarFreespaceChannel.

        Args:
            radar_target: An instance of a radar target providing target data.
            monostatic (bool, optional): True for monostatic configuration (default).
            rx_position (tf.Tensor, optional): Receiver position tensor with shape [batch_size, 3]
                                               for bistatic configuration.
            carrier_frequency (float, optional): Radar carrier frequency in Hz (default is 76.5e9).
        """
        self.radar_target = radar_target
        self.monostatic = monostatic
        self.rx_position = rx_position
        self.carrier_frequency = carrier_frequency
        self.wavelength = SPEED_OF_LIGHT / self.carrier_frequency  # λ = c / f

        if not self.monostatic and self.rx_position is None:
            raise ValueError("For bistatic configuration, rx_position must be provided.")

    def __call__(self, batch_size, num_time_steps, sampling_frequency):
        """
        Simulate the free-space channel.

        The channel coefficient 'a' is computed using the free-space attenuation model,
        incorporating the effect of the target's Radar Cross Section (RCS). For the monostatic
        configuration, the distance is computed from the transmitter (assumed at the origin) to
        the target. For the bistatic configuration, the attenuation is computed using both the
        transmitter-to-target and target-to-receiver distances. A phase shift corresponding to the
        propagation delay is applied.

        Args:
            batch_size (int): Number of samples (targets) in the batch.
            num_time_steps (int): Number of time steps in the simulation.
            sampling_frequency (float): Sampling frequency in Hz (unused in this simple model).

        Returns:
            a (tf.Tensor): Channel coefficients with shape
                           [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps],
                           where the output is computed for a single TX and single RX.
            tau (tf.Tensor): Propagation delays with shape [batch_size, num_rx, num_tx, num_paths] in seconds.
        """
        # Get target position from the radar target (shape: [batch_size, 3])
        target_pos = self.radar_target.position

        if self.monostatic:
            # Monostatic: Transmitter and receiver are co-located at the origin.
            R = tf.norm(target_pos, axis=-1)  # One-way distance [batch_size]
            # Free-space attenuation: (λ / (4πR))^2
            attenuation = (self.wavelength / (4 * PI * R)) ** 2
            # Phase shift: exp(-j * 2πR / λ)
            phase = tf.exp(tf.complex(tf.zeros_like(R), -2 * PI * R / self.wavelength))
            # Propagation delay (one-way delay)
            tau_val = R / SPEED_OF_LIGHT
        else:
            # Bistatic: Transmitter is at the origin and receiver is at rx_position.
            # Compute distance from transmitter to target.
            d_tx = tf.norm(target_pos, axis=-1)  # [batch_size]
            # Compute distance from target to receiver.
            rx_pos = self.rx_position  # [batch_size, 3]
            d_rx = tf.norm(target_pos - rx_pos, axis=-1)  # [batch_size]
            # Free-space attenuation: product of one-way attenuations.
            attenuation = (self.wavelength / (4 * PI * d_tx)) * (self.wavelength / (4 * PI * d_rx))
            # Phase shift: exp(-j * 2π*(d_tx+d_rx) / λ)
            phase = tf.exp(tf.complex(tf.zeros_like(d_tx + d_rx), -2 * PI * (d_tx + d_rx) / self.wavelength))
            # Propagation delay: sum of one-way delays.
            tau_val = (d_tx + d_rx) / SPEED_OF_LIGHT

        # Compute target RCS (in dBsm) and convert to linear scale.
        # For bistatic mode, the transmitter-to-target distance is used for RCS computation.
        distance_for_rcs = R if self.monostatic else d_tx
        rcs_db = self.radar_target.rcs(distance_for_rcs)
        rcs_lin = tf.pow(10.0, rcs_db / 10.0)

        # Compute the channel coefficient 'a'
        a_val = attenuation * tf.sqrt(rcs_lin) * phase  # [batch_size]

        # Reshape a_val to the expected dimensions:
        # [batch_size, num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=1, num_paths=1, num_time_steps]
        a_val = tf.reshape(a_val, [batch_size, 1, 1, 1, 1, 1])
        a_val = tf.tile(a_val, [1, 1, 1, 1, 1, num_time_steps])

        # Reshape tau to [batch_size, num_rx=1, num_tx=1, num_paths=1]
        tau_tensor = tf.reshape(tau_val, [batch_size, 1, 1, 1])

        return a_val, tau_tensor


if __name__ == "__main__":
    # Example usage of the RadarFreespaceChannel model.
    # For demonstration, import an example target (e.g., 'CarTarget') from sionna.channel.radar.target.
    from sionna.channel.radar.target import CarTarget  # Example target class (e.g., a car target)

    # Simulation parameters.
    batch_size = 4
    num_time_steps = 128
    sampling_frequency = 1e6  # Example: 1 MHz sampling frequency

    # Create sample positions and velocities for the target (assumed 3D coordinates).
    sample_positions = tf.constant([[50.0, 0.0, 0.0],
                                     [100.0, 0.0, 0.0],
                                     [75.0, 0.0, 0.0],
                                     [120.0, 0.0, 0.0]], dtype=tf.float32)
    sample_velocities = tf.constant([[10.0, 0.0, 0.0],
                                      [20.0, 0.0, 0.0],
                                      [15.0, 0.0, 0.0],
                                      [25.0, 0.0, 0.0]], dtype=tf.float32)

    # Instantiate a radar target using CarTarget (a car target example).
    target_instance = CarTarget(position=sample_positions, velocity=sample_velocities)

    # Example 1: Monostatic configuration (transmitter and receiver at the same location).
    mono_channel = RadarFreespaceChannel(radar_target=target_instance, monostatic=True)
    a_mono, tau_mono = mono_channel(batch_size, num_time_steps, sampling_frequency)
    print("Monostatic channel coefficient shape:", a_mono.shape)
    print("Monostatic delay shape:", tau_mono.shape)

    # Example 2: Bistatic configuration.
    # Define a receiver position for the bistatic case (e.g., receiver at [10, 0, 0] for each target).
    rx_positions = tf.constant([[10.0, 0.0, 0.0]] * batch_size, dtype=tf.float32)
    bi_channel = RadarFreespaceChannel(radar_target=target_instance, monostatic=False, rx_position=rx_positions)
    a_bi, tau_bi = bi_channel(batch_size, num_time_steps, sampling_frequency)
    print("Bistatic channel coefficient shape:", a_bi.shape)
    print("Bistatic delay shape:", tau_bi.shape)
