"""  Attention and normalization modules  """

from onmt.modules.global_attention import GlobalAttention
from onmt.modules.copy_generator import CopyGenerator, CopyGeneratorLoss, CopyGeneratorLossCompute

__all__ = ["GlobalAttention", "CopyGenerator",
           "CopyGeneratorLoss", "CopyGeneratorLossCompute", "sparse_activations"]
