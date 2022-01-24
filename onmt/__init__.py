""" Main entry point of the ONMT library """
from __future__ import division, print_function

import onmt.inputters
import onmt.encoders
import onmt.decoders
import onmt.models
import onmt.utils
import onmt.modules
import sys

# For Flake
__all__ = [onmt.inputters, onmt.encoders, onmt.decoders, onmt.models,
           onmt.utils, onmt.modules]

__version__ = "0.6.0"
