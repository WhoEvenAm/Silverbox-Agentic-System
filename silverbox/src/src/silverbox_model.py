"""Neural acceleration model, RK4 step, and rollout for Task 5.

y = centered output voltage (the oscillator's position)
v = dy/dt (velocity)
u = centered input voltage driving the oscillator

The network predicts acceleration, not position. RK4 uses that acceleration
and the current velocity to calculate the next position and velocity.
Training steps and the optimizer will belong in a separate training loop.
"""

import torch
import torch.nn as nn


# The homework's sampling frequency is 610.35 samples per second.
# DT is the time between samples, in seconds; it is not a training step count.
DT = 1.0 / 610.35


class AccelerationModel(nn.Module):
    """Predict acceleration from inputs ordered as (position, velocity, voltage)."""

    def __init__(
        self,
        hidden_sizes=(32, 32),  # Two hidden layers, each with 32 neurons.
        input_scale=(1.0, 1.0, 1.0),  # Divisors for (y, v, u); ones do no scaling / Normalization.
        acceleration_scale=1.0,  # Multiplier for the network's output.
    ):
        # Set up nn.Module so PyTorch can track our layers and parameters.
        super().__init__()

        # Convert Python settings to tensors matching the default weight dtype.
        scales = torch.tensor(input_scale, dtype=torch.float32)
        if (
            scales.shape != (3,)  # A vector with one scale per input: y, v, u.
            or not torch.isfinite(scales).all()  # Reject infinity and NaN.
            or not (scales > 0).all()  # Require positive, nonzero divisors.
        ):
            raise ValueError("input_scale must contain three finite positive values")

        output_scale = torch.tensor(acceleration_scale, dtype=torch.float32)
        if (
            output_scale.ndim != 0  # One scalar, rather than a vector or table.
            or not torch.isfinite(output_scale)  # The scalar must be finite.
            or output_scale <= 0  # It must also be positive.
        ):
            raise ValueError("acceleration_scale must be a finite positive scalar")

        # Require at least one hidden layer; every width is a neuron count.
        # any(...) is true if even one width fails either check.
        if not hidden_sizes or any(
            not isinstance(width, int) or width <= 0 for width in hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain positive integer widths")

        # Buffers are fixed tensors belonging to the model, not trained weights.
        # They are saved with the model and follow its device and dtype.
        self.register_buffer("input_scale", scales)
        self.register_buffer("acceleration_scale", output_scale)

        # Build the network as a list, then connect the layers in order.
        layers = []
        in_features = 3  # Each input sample has three values: (y, v, u).
        for width in hidden_sizes:
            # Linear learns weights and biases; Tanh adds nonlinearity.
            layers.extend([nn.Linear(in_features, width), nn.Tanh()])
            # This layer's output width is the next layer's input width.
            in_features = width

        # One output per sample. No final Tanh: acceleration is not bounded to +/-1.
        layers.append(nn.Linear(in_features, 1))
        # *layers unpacks the list; Sequential runs every layer in order.
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Calculate acceleration; PyTorch calls this when we write model(x).

        For N samples, x has shape (N, 3) and the output has shape (N, 1).
        A single vector of shape (3,) also works and returns shape (1,).
        """
        scaled_inputs = x / self.input_scale  # Divide y, v, u by their own scales.
        network_output = self.net(scaled_inputs)  # One value per input sample.
        acceleration = self.acceleration_scale * network_output
        return acceleration

    def state_derivative(self, state, u):
        """Return (dy/dt, dv/dt) = (v, alpha_theta(y, v, u)).

        For N samples, state has shape (N, 2) and u has shape (N, 1).
        The returned derivative has shape (N, 2), just like the state.
        Single vectors of shapes (2,) and (1,) also work.
        """
        # Join [y, v] with [u] to form the network input [y, v, u].
        # dim=-1 means join along the last dimension (the values per sample).
        inputs = torch.cat((state, u), dim=-1)
        # Calling self(...) invokes forward(), including both scaling operations.
        acceleration = self(inputs)
        # State index 0 is y; index 1 is v. The slice keeps a length-one dimension.
        # ... preserves preceding dimensions, such as the batch dimension.
        velocity = state[..., 1:2]
        # Return rates of change [dy/dt, dv/dt], not the next state [y, v].
        return torch.cat((velocity, acceleration), dim=-1)

    def rk4_step(self, state, u_now, u_next, dt=DT):
        """Advance the state one sample, interpolating voltage linearly.

        state is [y, v]. u_now and u_next are voltages at neighboring samples.
        dt is the time interval in seconds, defaulting to the sampling interval.
        This step remains differentiable so rollout errors can train the network.
        It uses the input voltage and current state, without measured outputs.
        """
        # Assume the voltage changes linearly between the two input samples.
        u_mid = (u_now + u_next) / 2

        # Each k is a pair of rates [velocity, acceleration].
        # k1: rates at the current state and time.
        k1 = self.state_derivative(state, u_now)
        # k2: rates at a midpoint state estimated using k1.
        k2 = self.state_derivative(state + dt * k1 / 2, u_mid)
        # k3: rates at a revised midpoint state estimated using k2.
        k3 = self.state_derivative(state + dt * k2 / 2, u_mid)
        # k4: rates at an endpoint state estimated using k3.
        k4 = self.state_derivative(state + dt * k3, u_next)

        # RK4's weighted average of rates, multiplied by dt, gives a state change.
        # Add that change to return the next [y, v].
        return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def rollout(self, initial_state, u, dt=DT):
        """Predict a sequence of states by repeatedly taking RK4 steps.

        For one sequence:
            initial_state: shape (2,), containing [y0, v0].
            u: shape (T, 1), containing T input voltage samples in time order.
            return: shape (T, 2), containing [predicted y, predicted v].

        For B sequences trained together, the shapes are (B, 2), (T, B, 1),
        and (T, B, 2), respectively. Time is always the first dimension of u.

        Row 0 is the initial state, so predictions line up with input samples.
        Measured output y is not an argument: later states come from predictions.
        Inputs should be tensors with the same dtype and device as the model.
        """
        # Each state must contain exactly two values: position and velocity.
        if initial_state.ndim < 1 or initial_state.shape[-1] != 2:
            raise ValueError("initial_state must have last dimension 2: [y0, v0]")

        # Replace the state's two values with one voltage value per sequence.
        # For example, state shape (B, 2) needs voltages of shape (T, B, 1).
        expected_input_shape = initial_state.shape[:-1] + (1,)
        if (
            u.ndim != initial_state.ndim + 1
            or u.shape[1:] != expected_input_shape
            or u.shape[0] == 0
        ):
            raise ValueError("u must contain at least one time sample, matching the state batch shape, with last dimension 1")

        state = initial_state
        predicted_states = [state]  # Sample 0 already has its initial state.

        # T samples have T-1 intervals; every step uses two neighboring voltages.
        for step in range(u.shape[0] - 1):
            state = self.rk4_step(state, u[step], u[step + 1], dt)
            predicted_states.append(state)

        # Stack the list into a tensor with time on dimension 0.
        # Do not detach states: training needs gradients through the whole rollout.
        # To obtain only predicted positions, select result[..., 0].
        return torch.stack(predicted_states, dim=0)
