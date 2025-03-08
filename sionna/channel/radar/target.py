import tensorflow as tf
import math

class RadarTarget(tf.Module):
    """
    Base class for radar targets.

    This class represents a generic radar target with a position and velocity.
    Both position and velocity are expected to be tensors with shape [batch_size, 3],
    representing 3D coordinates.

    Attributes:
        _position (tf.Tensor): The position of the target.
        _velocity (tf.Tensor): The velocity of the target.
    """
    def __init__(self, position, velocity, name=None):
        """
        Initialize a RadarTarget instance.

        Args:
            position: A tensor-like object representing the target's position.
            velocity: A tensor-like object representing the target's velocity.
            name (str, optional): The name of the module.
        """
        super().__init__(name=name)
        self._position = tf.convert_to_tensor(position, dtype=tf.float32)
        self._velocity = tf.convert_to_tensor(velocity, dtype=tf.float32)

    @property
    def position(self):
        """
        Get the target's position.

        Returns:
            tf.Tensor: The position tensor with shape [batch_size, 3].
        """
        return self._position

    @position.setter
    def position(self, new_position):
        """
        Set a new position for the target.

        Args:
            new_position: A tensor-like object representing the new position.
        """
        self._position = tf.convert_to_tensor(new_position, dtype=tf.float32)

    @property
    def velocity(self):
        """
        Get the target's velocity.

        Returns:
            tf.Tensor: The velocity tensor with shape [batch_size, 3].
        """
        return self._velocity

    @velocity.setter
    def velocity(self, new_velocity):
        """
        Set a new velocity for the target.

        Args:
            new_velocity: A tensor-like object representing the new velocity.
        """
        self._velocity = tf.convert_to_tensor(new_velocity, dtype=tf.float32)

    def rcs(self, R):
        """
        Calculate the Radar Cross Section (RCS) in dBsm for a given range R.

        This method must be implemented by subclasses.

        Args:
            R (tf.Tensor): A tensor containing the range(s) to the target.

        Returns:
            tf.Tensor: The calculated RCS in dBsm.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("The rcs() method must be implemented in the subclass.")


class CarTarget(RadarTarget):
    """
    Radar target representing a car.

    The RCS for the car is calculated using the empirical formula:
        RCScar = 20 + 5 * log10(max(10, R))
    This ensures that for ranges smaller than 10 meters, a minimum of 10 is used
    to avoid very low values.
    """
    def __init__(self, position, velocity, name=None):
        """
        Initialize a CarTarget instance.

        Args:
            position: A tensor-like object representing the car's position.
            velocity: A tensor-like object representing the car's velocity.
            name (str, optional): The name of the module.
        """
        super().__init__(position, velocity, name=name)

    def rcs(self, R):
        """
        Calculate the RCS for a car in dBsm.

        Args:
            R (tf.Tensor): A tensor containing the range(s) to the car.

        Returns:
            tf.Tensor: The calculated RCS in dBsm.
        """
        R_safe = tf.maximum(R, 10.0)
        rcs_db = 20.0 + 5.0 * (tf.math.log(R_safe) / tf.math.log(10.0))
        return rcs_db


class PedestrianTarget(RadarTarget):
    """
    Radar target representing a pedestrian.

    The RCS for the pedestrian is fixed at -10 dBsm.
    """
    def __init__(self, position, velocity, name=None):
        """
        Initialize a PedestrianTarget instance.

        Args:
            position: A tensor-like object representing the pedestrian's position.
            velocity: A tensor-like object representing the pedestrian's velocity.
            name (str, optional): The name of the module.
        """
        super().__init__(position, velocity, name=name)

    def rcs(self, R):
        """
        Return a fixed RCS value for a pedestrian.

        Args:
            R (tf.Tensor): A tensor containing the range(s) to the pedestrian (unused).

        Returns:
            tf.Tensor: A tensor filled with -10 dBsm, matching the shape of R.
        """
        return tf.fill(tf.shape(R), -10.0)


class MotorcycleTarget(RadarTarget):
    """
    Radar target representing a motorcycle.

    The RCS for the motorcycle is fixed at 7 dBsm.
    """
    def __init__(self, position, velocity, name=None):
        """
        Initialize a MotorcycleTarget instance.

        Args:
            position: A tensor-like object representing the motorcycle's position.
            velocity: A tensor-like object representing the motorcycle's velocity.
            name (str, optional): The name of the module.
        """
        super().__init__(position, velocity, name=name)

    def rcs(self, R):
        """
        Return a fixed RCS value for a motorcycle.

        Args:
            R (tf.Tensor): A tensor containing the range(s) to the motorcycle (unused).

        Returns:
            tf.Tensor: A tensor filled with 7 dBsm, matching the shape of R.
        """
        return tf.fill(tf.shape(R), 7.0)


class TruckTarget(RadarTarget):
    """
    Radar target representing a truck.

    The RCS for the truck is calculated using the empirical formula:
        RCSTruck = 45 + 5 * log10(max(10, R))
    This formula is used to account for the dependence of the RCS on the distance R.
    """
    def __init__(self, position, velocity, name=None):
        """
        Initialize a TruckTarget instance.

        Args:
            position: A tensor-like object representing the truck's position.
            velocity: A tensor-like object representing the truck's velocity.
            name (str, optional): The name of the module.
        """
        super().__init__(position, velocity, name=name)

    def rcs(self, R):
        """
        Calculate the RCS for a truck in dBsm.

        Args:
            R (tf.Tensor): A tensor containing the range(s) to the truck.

        Returns:
            tf.Tensor: The calculated RCS in dBsm.
        """
        R_safe = tf.maximum(R, 10.0)
        rcs_db = 45.0 + 5.0 * (tf.math.log(R_safe) / tf.math.log(10.0))
        return rcs_db


if __name__ == "__main__":
    # Demonstration of how to use the RadarTarget subclasses.

    # Define a batch size for demonstration purposes.
    batch_size = 4

    # Sample positions (in meters) and velocities (in m/s) for a batch of targets.
    sample_positions = tf.constant([[50.0, 0.0, 0.0],
                                    [100.0, 0.0, 0.0],
                                    [75.0, 0.0, 0.0],
                                    [120.0, 0.0, 0.0]], dtype=tf.float32)
    sample_velocities = tf.constant([[10.0, 0.0, 0.0],
                                     [20.0, 0.0, 0.0],
                                     [15.0, 0.0, 0.0],
                                     [25.0, 0.0, 0.0]], dtype=tf.float32)

    # Create instances of each target type.
    car_target = CarTarget(position=sample_positions, velocity=sample_velocities)
    pedestrian_target = PedestrianTarget(position=sample_positions, velocity=sample_velocities)
    motorcycle_target = MotorcycleTarget(position=sample_positions, velocity=sample_velocities)
    truck_target = TruckTarget(position=sample_positions, velocity=sample_velocities)

    # Define a sample range tensor (in meters) for which to calculate the RCS.
    sample_range = tf.constant([50.0, 100.0, 75.0, 120.0], dtype=tf.float32)

    # Calculate and print RCS for each target type.
    print("Car Target RCS (dBsm):")
    print(car_target.rcs(sample_range))

    print("\nPedestrian Target RCS (dBsm):")
    print(pedestrian_target.rcs(sample_range))

    print("\nMotorcycle Target RCS (dBsm):")
    print(motorcycle_target.rcs(sample_range))

    print("\nTruck Target RCS (dBsm):")
    print(truck_target.rcs(sample_range))
