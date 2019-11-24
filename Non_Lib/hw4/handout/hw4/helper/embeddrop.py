"""Module containing the shared RNN model."""
import numpy as np
import collections

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


def _get_dropped_weights(w_raw, dropout_p, is_training):
    """Drops out weights to implement DropConnect.
    Args:
        w_raw: Full, pre-dropout, weights to be dropped out.
        dropout_p: Proportion of weights to drop out.
        is_training: True iff _shared_ model is training.
    Returns:
        The dropped weights.
    TODO(brendan): Why does torch.nn.functional.dropout() return:
    1. `torch.autograd.Variable()` on the training loop
    2. `torch.nn.Parameter()` on the controller or eval loop, when
    training = False...
    Even though the call to `_setweights` in the Smerity repo's
    `weight_drop.py` does not have this behaviour, and `F.dropout` always
    returns `torch.autograd.Variable` there, even when `training=False`?
    The above TODO is the reason for the hacky check for `torch.nn.Parameter`.
    """
    dropped_w = F.dropout(w_raw, p=dropout_p, training=is_training)

    if isinstance(dropped_w, torch.nn.Parameter):
        dropped_w = dropped_w.clone()

    return dropped_w


def isnan(tensor):
    return np.isnan(tensor.cpu().data.numpy()).sum() > 0


class EmbeddingDropout(torch.nn.Embedding):
    """Class for dropping out embeddings by zero'ing out parameters in the
    embedding matrix.
    This is equivalent to dropping out particular words, e.g., in the sentence
    'the quick brown fox jumps over the lazy dog', dropping out 'the' would
    lead to the sentence '### quick brown fox jumps over ### lazy dog' (in the
    embedding vector space).
    See 'A Theoretically Grounded Application of Dropout in Recurrent Neural
    Networks', (Gal and Ghahramani, 2016).
    """
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 max_norm=None,
                 norm_type=2,
                 scale_grad_by_freq=False,
                 sparse=False,
                 dropout=0.4,
                 scale=None):
        """Embedding constructor.
        Args:
            dropout: Dropout probability.
            scale: Used to scale parameters of embedding weight matrix that are
                not dropped out. Note that this is _in addition_ to the
                `1/(1 - dropout)` scaling.
        See `torch.nn.Embedding` for remaining arguments.
        """
        torch.nn.Embedding.__init__(self,
                                    num_embeddings=num_embeddings,
                                    embedding_dim=embedding_dim,
                                    max_norm=max_norm,
                                    norm_type=norm_type,
                                    scale_grad_by_freq=scale_grad_by_freq,
                                    sparse=sparse)
        self.dropout = dropout
        assert (dropout >= 0.0) and (dropout < 1.0), ('Dropout must be >= 0.0 '
                                                      'and < 1.0')
        self.scale = scale

    def forward(self, inputs):  # pylint:disable=arguments-differ
        """Embeds `inputs` with the dropped out embedding weight matrix."""
        if self.training:
            dropout = self.dropout
        else:
            dropout = 0

        if dropout:
            mask = self.weight.data.new(self.weight.size(0), 1)
            mask.bernoulli_(1 - dropout)
            mask = mask.expand_as(self.weight)
            mask = mask / (1 - dropout)
            masked_weight = self.weight * Variable(mask)
        else:
            masked_weight = self.weight
        if self.scale and self.scale != 1:
            masked_weight = masked_weight * self.scale

        return F.embedding(inputs,
                           masked_weight,
                           max_norm=self.max_norm,
                           norm_type=self.norm_type,
                           scale_grad_by_freq=self.scale_grad_by_freq,
                           sparse=self.sparse)