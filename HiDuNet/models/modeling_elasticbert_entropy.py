# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch ElasticBERT model for Early Exit with Entropy. """

import math
import sys

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import LayerNorm
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.functional import softplus
from torch.distributions import Normal, Independent

from transformers.activations import ACT2FN
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,   
)
from transformers.utils import logging
from .configuration_elasticbert import ElasticBertConfig
from torch.nn.functional import softplus


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "fnlp/elasticbert-base"
_CONFIG_FOR_DOC = "ElasticBertConfig"
_TOKENIZER_FOR_DOC = "ElasticBertTokenizer"


ELASTICBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "fnlp/elasticbert-base",
    "fnlp/elasticbert-large",
]


class GradientRescaleFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input)
        ctx.gd_scale_weight = weight
        output = input
        return output
    
    @staticmethod
    def backward(ctx, grad_outputs):
        input = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = ctx.gd_scale_weight * grad_outputs

        return grad_input, grad_weight

gradient_rescale = GradientRescaleFunction.apply


class ElasticBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
            padding_idx=config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, 
            config.hidden_size, 
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, 
            config.hidden_size,
        )

        # self.layernorm is not snake-cased to stick with TensorFlow model 
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(
            config.hidden_size, 
            eps=config.layer_norm_eps, 
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and 
        # exported when serialized
        self.register_buffer(
            "position_ids", 
            torch.arange(config.max_position_embeddings).expand((1, -1)) 
        )
        self.position_embedding_type = getattr(
            config, 
            "position_embedding_type", 
            "absolute",
        )

    def forward(
        self, 
        input_ids=None, 
        token_type_ids=None, 
        position_ids=None, 
        inputs_embeds=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, 
                dtype=torch.long, 
                device=self.position_ids.device,
            )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ElasticBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 \
            and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple"\
                "of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = \
            config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(
            config, 
            "position_embedding_type", 
            "absolute",
        )
        if self.position_embedding_type == "relative_key" or \
           self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, 
                self.attention_head_size,
            )

    def transpose_for_scores(self, x):
        new_x_shape = \
            x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" 
        # to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or \
           self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, 
                dtype=torch.long, 
                device=hidden_states.device,
            ).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, 
                dtype=torch.long,
                device=hidden_states.device,
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            # fp16 compatibility
            positional_embedding = \
                positional_embedding.to(dtype=query_layer.dtype)

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", 
                    query_layer, 
                    positional_embedding,
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", 
                    query_layer, 
                    positional_embedding,
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", 
                    key_layer, 
                    positional_embedding
                )
                attention_scores = \
                    attention_scores + relative_position_scores_query + \
                    relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in 
            # ElasticBertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions \
            else (context_layer,)

        return outputs


class ElasticBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ElasticBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = ElasticBertSelfAttention(config)
        self.output = ElasticBertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, 
            self.self.num_attention_heads, 
            self.self.attention_head_size, 
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = \
            self.self.num_attention_heads - len(heads)
        self.self.all_head_size = \
            self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class ElasticBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ElasticBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ElasticBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ElasticBertAttention(config)
        self.intermediate = ElasticBertIntermediate(config)
        self.output = ElasticBertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):

        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        # add self attentions if we output attention weights
        outputs = self_attention_outputs[1:]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, 
            self.chunk_size_feed_forward, 
            self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class ElasticBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()


    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ElasticBertEncoder(nn.Module):
    def __init__(self, config, add_pooling_layer=None):
        super().__init__()
        self.config = config
        self.add_pooling_layer = add_pooling_layer
        self.num_output_layers = config.num_output_layers
        self.num_hidden_layers = config.num_hidden_layers
        self.max_output_layers = config.max_output_layers

        self.layer = nn.ModuleList([ElasticBertLayer(config) \
            for _ in range(config.num_hidden_layers)])

        assert self.num_output_layers <= self.num_hidden_layers, \
            "The total number of layers must be be greater than or "\
            "equal to the number of the output layers. "
        
        self.pooler = nn.ModuleList([ElasticBertPooler(config) \
            for i in range(self.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        final_pooled_output = None
        output_sequence_outputs = None
        output_pooled_outputs = None

        for i, layer_module in enumerate(self.layer):

            if getattr(self.config, "gradient_checkpointing", False) \
                and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]

            if i == self.num_hidden_layers - 1:
                if self.add_pooling_layer:
                    final_pooled_output = \
                        self.pooler[11](hidden_states)
                else:
                    final_pooled_output = hidden_states[:, 0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)


        return tuple(
                v
                for v in [
                    hidden_states,
                    output_sequence_outputs,
                    output_pooled_outputs,
                    final_pooled_output,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
        )

    def adaptive_forward(
        self,
        hidden_states=None,
        current_layer=None, 
        attention_mask=None,
    ):

        layer_outputs = self.layer[current_layer](
            hidden_states,
            attention_mask,
            output_attentions=False,            
        )

        hidden_states = layer_outputs[0]  
        
        if self.training:
            hidden_states = gradient_rescale(
                hidden_states, 1.0 / (self.num_hidden_layers - current_layer)
            ) 

        pooled_output = None
        if self.add_pooling_layer:
            pooled_output = self.pooler[current_layer](
                hidden_states,
            )
        
        return hidden_states, pooled_output


class ElasticBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface 
    for downloading and loading pretrained models.
    """

    config_class = ElasticBertConfig
    base_model_prefix = "elasticbert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal
            # for initialization cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, 
                std=self.config.initializer_range,
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, 
                std=self.config.initializer_range,
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)           


ELASTICBERT_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)
    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.
    Parameters:
        config (:class:`~ElasticBertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""


ELASTICBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.
            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:
            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.
            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.
            `What are position IDs? <../glossary.html#position-ids>`_
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
"""           


@add_start_docstrings(
    "The bare ElasticBert Model transformer outputting raw hidden-states without any specific head on top.",
    ELASTICBERT_START_DOCSTRING,    
)
class ElasticBertModel(ElasticBertPreTrainedModel):

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.add_pooling_layer = add_pooling_layer
        self.num_output_layers = config.num_output_layers
        self.num_hidden_layers = config.num_hidden_layers
        self.max_output_layers = config.max_output_layers

        self.embeddings = ElasticBertEmbeddings(config)
        self.encoder = ElasticBertEncoder(
            config,
            add_pooling_layer=add_pooling_layer,
        )

        self.init_weights()

        self.eval_highway = False
        self.inference_instances_num = 0
        self.inference_layers_num = 0
        self.exiting_layer_every_ins = []


    @add_start_docstrings_to_model_forward(
        ELASTICBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        config_class=_CONFIG_FOR_DOC,        
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_dropout=None,
        output_layers=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        output_attentions = output_attentions if output_attentions is not None \
            else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None \
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and "\
                "inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds"
            )

        device = input_ids.device if input_ids is not None else \
            inputs_embeds.device


        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length)), 
                device=device,
            )
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, 
                dtype=torch.long, 
                device=device,
            )

        # We can provide a self-attention mask of dimensions 
        # [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable 
        # to all heads.
        extended_attention_mask: torch.Tensor = \
            self.get_extended_attention_mask(attention_mask, input_shape, device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

    
        encoder_outputs = embedding_output
        res = []
        mid_fea = []
        for i in range(self.num_hidden_layers):
            encoder_outputs, pooled_output=self.encoder.adaptive_forward(
                encoder_outputs,
                current_layer=i,
                attention_mask=extended_attention_mask,
            )
            logits = None
            mid_fea.append(pooled_output)
            if i<self.num_output_layers and self.add_pooling_layer:
                assert pooled_output is not None
                logits = output_layers[i](output_dropout(pooled_output))

            encoder_outputs = gradient_rescale(
                encoder_outputs, 
                (self.num_hidden_layers - i -1),
            )

            res.append(logits)                
        assert len(res) == self.num_hidden_layers and \
               len(mid_fea) == self.num_hidden_layers

        return res, mid_fea, embedding_output

@add_start_docstrings(
    """
    ElasticBert Model transformer with a sequence classification/regression head on top 
    (a linear layer on top of the pooled output) e.g. for GLUE tasks.
    """,
    ELASTICBERT_START_DOCSTRING,
)           


# VAE encoder 
class VAE_Encoder(nn.Module):
    def __init__(self, args, z_dim=128):
        super(VAE_Encoder, self).__init__()
        self.z_dim = z_dim
        self.linear = nn.Linear(z_dim, 1)
        self.net = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(True),
            nn.Linear(128, z_dim * 2),
        )

    def forward(self, x):
        x = self.linear(x.permute(0, 2, 1))
        params = self.net(x.squeeze(2))
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7
        return Independent(Normal(loc=mu, scale=sigma), 1)


# Statistic Feature Network
class S_Net(nn.Module):
    def __init__(self, args):
        super(S_Net, self).__init__()
        self.encoder = VAE_Encoder(args)

    def forward(self, inputs):
        p_z = self.encoder(inputs)
        z = p_z.rsample()
        return z



# intra attention
class IntraAttention(nn.Module):
    def __init__(self, d_model):
        super(IntraAttention, self).__init__()
        self.d_model = d_model
        self.query = nn.Linear(self.d_model, self.d_model)
        self.value = nn.Linear(self.d_model, self.d_model)
        self.key = nn.Linear(self.d_model, self.d_model)
        self.linear = nn.Linear(2*self.d_model, d_model)
        self.aoa = nn.Sequential(
            nn.Linear(2*self.d_model, 2*self.d_model),
            nn.GLU(),
        )
    
    def forward(self, input_features):
        Q = self.query(input_features)
        K = self.key(input_features)
        V = self.value(input_features)
        scores = torch.matmul(Q, K.transpose(-2, -1)) \
             / math.sqrt(self.d_model)
        alpha = torch.softmax(scores, dim=1)
        V_ = torch.matmul(alpha, V)
        x = torch.cat([V_, Q], dim=1)
        x1 = self.linear(x)
        alpha_ = self.aoa(x)
        x = torch.mul(x1, alpha_)

        return x




# Adaptive Feature Fusion Network
class FusionNet(nn.Module):
    def __init__(self, shared_dim=128):
        super(FusionNet, self).__init__()
        self.Sen_aligner = nn.Sequential(
            nn.Linear(768, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
        )
        self.Sat_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
        )
        self.Sen_intra_att = IntraAttention(shared_dim)
        self.Sat_intra_att = IntraAttention(shared_dim)
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(shared_dim, 2)


    def forward(self, Sen_features, Sat_features):
        Sen_features = self.Sen_aligner(Sen_features)
        Sat_features = self.Sat_aligner(Sat_features)

        Sen_Att_features = self.Sen_intra_att(Sen_features)
        Sat_Att_features = self.Sat_intra_att(Sat_features)

        # Fused_features = torch.cat([Sen_Att_features, Sat_Att_features], dim=1)
        Fused_features = torch.add(Sen_Att_features, Sat_Att_features)
        

        logits = self.classifier(self.dropout(Fused_features))
        return logits


class ElasticBertForSequenceClassification(ElasticBertPreTrainedModel):
    def __init__(self, config, args, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.add_pooling_layer = add_pooling_layer

        self.bert = ElasticBertModel(
            config, add_pooling_layer=add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifiers = nn.ModuleList(
            [nn.Linear(config.hidden_size, self.config.num_labels) \
            for _ in range(config.num_output_layers)])

        self.init_weights()
        self.args = args

        self.s_net = S_Net(self.args)
        self.fusion_net = FusionNet()

    @add_start_docstrings_to_model_forward(
        ELASTICBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        config_class=_CONFIG_FOR_DOC,
    )  
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
    
        H_logits, H_features, embedding = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_dropout=self.dropout,
            output_layers=self.classifiers,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        H_total_loss = None
        for i in range(len(self.classifiers)):
            H_loss_fct = CrossEntropyLoss()
            H_loss = H_loss_fct(
                H_logits[i].view(-1, self.num_labels), 
                labels.view(-1),
            )
            if H_total_loss is None:
                H_total_loss = H_loss
            else:
                H_total_loss += H_loss

        H_total_loss = H_total_loss / (len(self.classifiers))

        Sat_features = self.s_net(inputs=embedding)

        Sen_features = H_features[-1]

        logits = self.fusion_net(Sen_features, Sat_features)
        total_loss = None
        loss = None
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            logits.view(-1, self.num_labels), 
            labels.view(-1),
        )

        total_loss = (H_total_loss + loss) / 2 

        return (total_loss, logits)               

