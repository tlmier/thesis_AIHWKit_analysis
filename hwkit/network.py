"""The front end you edit to try a new experiment: network architecture +
training hyperparameters. Imported by simulation.py; run `python main.py`
to launch the GUI, not this file directly."""

import torch.nn as nn

from aihwkit.nn import AnalogConv2d, AnalogLinear, AnalogSequential


class FlattenAnalogSequential(AnalogSequential):
    '''Flattens 2D image input before feeding it to the wrapped layers.'''

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return super().forward(x)


class CNNAnalogSequential(AnalogSequential):
    '''Applies the convolutional layers to the (batch, channels, size, size)
    image input, flattens once the conv stack is done, then applies the
    fully-connected layers.'''

    def __init__(self, conv_layers, fc_layers):
        super().__init__(*conv_layers, *fc_layers)
        self._n_conv_modules = len(conv_layers)

    def forward(self, x):
        '''_summary_

        Args:
            x (any): image data of the image

        Returns:
            any: image forwarded trough the network
        '''
        for i, layer in enumerate(self):
            if i == self._n_conv_modules:
                x = x.reshape(x.size(0), -1)
            x = layer(x)
        return x


def mlp(rpu_config, args, input_channels=1, input_size=28):
    '''Build an MLP from args -- the dict gui_frontend.App.do_config_network()
    produces (and config_loading.load_network()/dump_network() round-trip
    to/from a .net file):

    Args:
        rpu_config (_type_): loaded from the file / file type
        args (ArrayLike): args to build the network
        input_channels (int, optional): input chanels (Color 3/ BW 1). Defaults to 1.
        input_size (int, optional): (nxn of the input). Defaults to 28.

    Returns:
        _type_: model of the System
    '''
    hidden_sizes = args.get("hidden_sizes", [256, 128])
    activation_cls = getattr(nn, args.get("activation", "ReLU"))

    sizes = [input_channels * input_size * input_size, *hidden_sizes, 10]
    layers = []
    for i in range(len(sizes) - 1):
        if rpu_config is None:
            # Digital baseline (the "Ideal" device mode): plain torch layers, exact floats.
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=True))
        else:
            layers.append(AnalogLinear(sizes[i], sizes[i + 1], bias=True, rpu_config=rpu_config))
        if i < len(sizes) - 2:
            layers.append(activation_cls())
    return FlattenAnalogSequential(*layers)


def cnn(rpu_config, args, input_channels=1, input_size=28):
    '''Build an CNN from args -- the dict gui_frontend.App.do_config_network()
    produces (and config_loading.load_network()/dump_network() round-trip
    to/from a .net file):

    Args:
        rpu_config (_type_): loaded from the file / file type
        args (ArrayLike): args to build the network
        input_channels (int, optional): input chanels (Color 3/ BW 1). Defaults to 1.
        input_size (int, optional): (nxn of the input). Defaults to 28.

    Raises:
        ValueError: due to no padding the convolution image gets smaller

    Returns:
        _type_: model of the system
    '''
    conv_specs = args.get(
        "conv_layers", [{"out_channels": 16, "kernel_size": 3}, {"out_channels": 32, "kernel_size": 3}],
    )
    hidden_sizes = args.get("hidden_sizes", [128])
    activation_cls = getattr(nn, args.get("activation", "ReLU"))

    conv_layers = []
    in_channels = input_channels
    spatial = input_size
    for spec in conv_specs:
        out_channels = spec["out_channels"]
        kernel_size = spec["kernel_size"]
        if rpu_config is None:
            conv_layers.append(nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=True,
            ))
        else:
            conv_layers.append(AnalogConv2d(
                in_channels, out_channels, kernel_size,
                padding=kernel_size // 2, bias=True, rpu_config=rpu_config,
            ))
        conv_layers.append(activation_cls())
        conv_layers.append(nn.MaxPool2d(2))
        in_channels = out_channels
        spatial //= 2
        if spatial < 1:
            raise ValueError(f"Too many conv layers for a {input_size}x{input_size} input (spatial size hit 0).")

    flat_size = in_channels * spatial * spatial
    sizes = [flat_size, *hidden_sizes, 10]
    fc_layers = []
    for i in range(len(sizes) - 1):
        if rpu_config is None:
            fc_layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=True))
        else:
            fc_layers.append(AnalogLinear(sizes[i], sizes[i + 1], bias=True, rpu_config=rpu_config))
        if i < len(sizes) - 2:
            fc_layers.append(activation_cls())

    return CNNAnalogSequential(conv_layers, fc_layers)


def build_model(rpu_config, net_type, args, input_channels=1, input_size=28):
    '''Defines the network architecture

    Args:
        rpu_config (_type_): RPU config for the analog layers, or None to build the
            purely digital software baseline (plain nn.Linear/nn.Conv2d, "Ideal" mode)
        net_type (str): type of the network
        args (ArrayLike): args to build the network
        input_channels (int, optional): input chanels (Color 3/ BW 1). Defaults to 1.
        input_size (int, optional): (nxn of the input). Defaults to 28.

    Raises:
        ValueError: Unknown network type

    Returns:
        _type_: Network
    '''
    if net_type == "MLP":
        return mlp(rpu_config, args, input_channels, input_size)
    if net_type == "CNN":
        return cnn(rpu_config, args, input_channels, input_size)
    raise ValueError(f"Unknown network type: {net_type!r}")
