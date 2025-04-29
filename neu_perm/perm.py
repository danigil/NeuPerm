import copy
from functools import partial
from typing import List, Literal, Optional, Tuple
import numpy as np
import torch
from collections import OrderedDict

norm_varnames = ['weight', 'bias', 'running_mean', 'running_var']

def get_perm_idxs(n_c):
    # return torch.flip(torch.arange(0, n_c, dtype=int), dims=[0])
    return torch.randperm(n_c, dtype=int)

def print_sd(sd):
    for k, v in sd.items():
        print(k, v.shape)
    return

def check(model, sd, sd_perm, batch_size=1, input_shape=(3, 224, 224)):
    model.eval()
    model = model.to('cpu')

    x = torch.randn((batch_size,)+input_shape).to('cpu')

    model.load_state_dict(sd)
    output_orig = model(x)

    model.load_state_dict(sd_perm)
    output_perm = model(x)

    isclose = np.allclose(output_orig.detach().numpy(), output_perm.detach().numpy(), rtol=1e-6, atol=1e-6)
    print("Allclose:", isclose)

    diff = np.abs(output_orig.detach().numpy() - output_perm.detach().numpy())
    print("Max absolute difference:", diff.max())
    print("Mean absolute difference:", diff.mean())

    return

VGG_11_ALL_FEATURES = [(0,3), (3,6), (6,8), (8,11), (11,13), (13,16), (16,18)]
VGG_11_ALL_MLPS = [(0,3), (3,6)]

VGG_16_FEATURES = [(0,2), (2,5), (5,7), (7,10), (10,12), (12,14), (14,17), (17,19), (19,21), (21,24), (24,26), (26,28)]
VGG_16_MLPS = [(0,3), (3,6)]

def vgg(
    sd:OrderedDict,
    inplace:bool=True,

    features:Optional[List[int]]=None,
    mlps:Optional[List[int]]=None,

    model_type:Literal['vgg11', 'vgg16']='vgg11',

    key_prefix:str='',
):
    def permute_layers(sd, layer1, layer2):
        layer1_w = f'{layer1}.weight'
        layer2_w = f'{layer2}.weight'
        
        layer1_b = f'{layer1}.bias'

        n_c_1 = sd[layer1_w].shape[0]
        idxs_1 = get_perm_idxs(n_c_1)

        sd[layer1_w] = sd[layer1_w][idxs_1,...]
        sd[layer1_b] = sd[layer1_b][idxs_1,...]
        sd[layer2_w] = sd[layer2_w][:,idxs_1,...]

        return
    
    if features is None:
        if model_type == 'vgg11':
            features = VGG_11_ALL_FEATURES
        elif model_type == 'vgg16':
            features = VGG_16_FEATURES
    
    if mlps is None:
        if model_type == 'vgg11':
            mlps = VGG_11_ALL_MLPS
        elif model_type == 'vgg16':
            mlps = VGG_16_MLPS

    if inplace:
        sd_perm = sd
    else:
        sd_perm = copy.deepcopy(sd)

    for ftr1, ftr2 in features:
        layer1 = f'{key_prefix}features.{ftr1}'
        layer2 = f'{key_prefix}features.{ftr2}'
        permute_layers(sd_perm, layer1, layer2)

    for mlp1, mlp2 in mlps:
        layer1 = f'{key_prefix}classifier.{mlp1}'
        layer2 = f'{key_prefix}classifier.{mlp2}'
        permute_layers(sd_perm, layer1, layer2)

    return sd_perm

vgg11 = partial(vgg, model_type='vgg11')
vgg16 = partial(vgg, model_type='vgg16')

DENSENET121_ALL_BLOCKS = \
    [(1, i) for i in range(1, 7)] + \
    [(2, i) for i in range(1, 13)] + \
    [(3, i) for i in range(1, 25)] + \
    [(4, i) for i in range(1, 17)]

def densenet121(
    sd:OrderedDict,
    inplace:bool=True,

    blocks:Optional[List[Tuple[int, int]]]=None,

    model_type:Literal['densenet121',]='densenet121',

    key_prefix:str='',
):
    def permute_denseblock(sd, block_idx, layer_idx):
        module_prefix = f'{key_prefix}features.denseblock{block_idx}.denselayer{layer_idx}'
        conv1 = f'{module_prefix}.conv1'
        bn2 = f'{module_prefix}.norm2'
        conv2 = f'{module_prefix}.conv2'

        permute_conv_bn_conv(sd, conv1, bn2, conv2)

    if blocks is None:
        blocks = DENSENET121_ALL_BLOCKS

    if inplace:
        sd_perm = sd
    else:
        sd_perm = copy.deepcopy(sd)

    for block_idx, layer_idx in blocks:
        permute_denseblock(sd_perm, block_idx, layer_idx)

    return sd_perm

RESNET50_ALL_BLOCKS = ['1.0', '1.1', '1.2', '2.0', '2.1', '2.2', '2.3', '3.0', '3.1', '3.2', '3.3', '3.4', '3.5', '4.0', '4.1', '4.2']
RESNET101_ALL_BLOCKS = ['1.0', '1.1', '1.2', '2.0', '2.1', '2.2', '2.3'] + [f'3.{i}' for i in range(0, 23)] + [f'4.{i}' for i in range(0, 3)]

def permute_conv_bn_conv(sd, conv1, bn, conv2):
    conv1_w = f'{conv1}.weight'
    conv1_b = f'{conv1}.bias'

    conv2_w = f'{conv2}.weight'

    n_c_1 = sd[conv1_w].shape[0]
    idxs_1 = get_perm_idxs(n_c_1)

    sd[conv1_w] = sd[conv1_w][idxs_1,...]
    if conv1_b in sd:
        sd[conv1_b] = sd[conv1_b][idxs_1,...]

    for varname in norm_varnames:
        sd[f'{bn}.{varname}'] = sd[f'{bn}.{varname}'][idxs_1,...]

    sd[conv2_w] = sd[conv2_w][:,idxs_1,...]

    return


def resnet(
    sd:OrderedDict,
    inplace:bool=True,

    blocks:Optional[List[str]]=None,

    model_type:Literal['resnet50', 'resnet101']='resnet50',

    key_prefix:str='',
):
    def permute_resblock(sd, block):
        block = f'{key_prefix}layer{block}'
        
        conv1 = f'{block}.conv1'
        bn1 = f'{block}.bn1'

        conv2 = f'{block}.conv2'
        bn2 = f'{block}.bn2'

        conv3 = f'{block}.conv3'
        bn3 = f'{block}.bn3'

        downsample_conv = f'{block}.downsample.0'
        downsample_bn = f'{block}.downsample.1'

        permute_conv_bn_conv(sd, conv1, bn1, conv2)
        permute_conv_bn_conv(sd, conv2, bn2, conv3)

        if downsample_conv in sd:
            permute_conv_bn_conv(sd, conv3, bn3, downsample_conv)

    if blocks is None:
        if model_type == 'resnet50':
            blocks = RESNET50_ALL_BLOCKS
        elif model_type == 'resnet101':
            blocks = RESNET101_ALL_BLOCKS

    if inplace:
        sd_perm = sd
    else:
        sd_perm = copy.deepcopy(sd)

    for block in blocks:
        permute_resblock(sd_perm, block)

    return sd_perm

resnet50 = partial(resnet, model_type='resnet50')
resnet101 = partial(resnet, model_type='resnet101')

"""
    LLMs
"""
LLAMA3_2_1B_ALL_BLOCKS = list(range(0,16))

def llama(
    sd:OrderedDict,
    inplace:bool=True,

    blocks:Optional[List[str]]=None,

    model_type:Literal['llama-3.2-1b']='llama-3.2-1b',

    key_prefix:str='',
):
    def permute_llama2_layer(
        sd,
        layer_name="model.layers.0",
        inplace:bool=False,
        num_heads=32,

        attn_perm:bool=True,
        mlp_perm:bool=True,

        gqa:bool=True,

    ):
        if inplace:
            sd_perm = sd
        else:
            sd_perm = copy.deepcopy(sd)

        q_proj_weight_key = f"{layer_name}.self_attn.q_proj.weight"
        k_proj_weight_key = f"{layer_name}.self_attn.k_proj.weight"
        v_proj_weight_key = f"{layer_name}.self_attn.v_proj.weight"
        o_proj_weight_key = f"{layer_name}.self_attn.o_proj.weight"

        embed_dim, _ = sd_perm[q_proj_weight_key].shape
        kv_dim, _ = sd_perm[k_proj_weight_key].shape
        head_dim = embed_dim // num_heads

        n_kv_heads = kv_dim // head_dim
        params_per_kv_head = kv_dim // n_kv_heads

        group_size = embed_dim // kv_dim

        if attn_perm:
            if gqa:
                kv_idxs = torch.arange(kv_dim, dtype=int).reshape((n_kv_heads, params_per_kv_head))
                kv_head_perm_idxs = get_perm_idxs(n_kv_heads)
                kv_perm_idxs = kv_idxs[kv_head_perm_idxs, :].reshape(-1)

                q_idxs = torch.arange(embed_dim, dtype=int).reshape((n_kv_heads, group_size, params_per_kv_head))
                q_idxs = q_idxs[kv_head_perm_idxs, :, :].reshape(-1)

                sd_perm[q_proj_weight_key] = sd_perm[q_proj_weight_key][q_idxs, ...]
                sd_perm[k_proj_weight_key] = sd_perm[k_proj_weight_key][kv_perm_idxs, ...]
                sd_perm[v_proj_weight_key] = sd_perm[v_proj_weight_key][kv_perm_idxs, ...]
                sd_perm[o_proj_weight_key] = sd_perm[o_proj_weight_key][..., q_idxs]

            else:
                idxs = torch.arange(embed_dim, dtype=int).reshape((num_heads, head_dim))
                head_perm_idxs = get_perm_idxs(num_heads)
                perm_idxs = idxs[head_perm_idxs, :].reshape(-1)
                
                sd_perm[q_proj_weight_key] = sd_perm[q_proj_weight_key][perm_idxs, ...]
                sd_perm[k_proj_weight_key] = sd_perm[k_proj_weight_key][perm_idxs, ...]
                sd_perm[v_proj_weight_key] = sd_perm[v_proj_weight_key][perm_idxs, ...]
                sd_perm[o_proj_weight_key] = sd_perm[o_proj_weight_key][..., perm_idxs]


        if mlp_perm:
            mlp_gate_weight_key = f"{layer_name}.mlp.gate_proj.weight"
            mlp_up_proj_weight_key = f"{layer_name}.mlp.up_proj.weight"
            mlp_down_proj_weight_key = f"{layer_name}.mlp.down_proj.weight"

            gate_hidden_dim, _ = sd_perm[mlp_gate_weight_key].shape
            perm_idxs_mlp = get_perm_idxs(gate_hidden_dim)
            sd_perm[mlp_gate_weight_key] = sd_perm[mlp_gate_weight_key][perm_idxs_mlp, ...]
            sd_perm[mlp_up_proj_weight_key] = sd_perm[mlp_up_proj_weight_key][perm_idxs_mlp, ...]
            sd_perm[mlp_down_proj_weight_key] = sd_perm[mlp_down_proj_weight_key][..., perm_idxs_mlp]

        return sd_perm

    def permute_llama2_all_layers(sd, blocks=LLAMA3_2_1B_ALL_BLOCKS, inplace:bool=False):
        if inplace:
            sd_perm = sd
        else:
            sd_perm = copy.deepcopy(sd)

        for i in blocks:
            layer_name = f"model.layers.{i}"
            permute_llama2_layer(sd_perm, layer_name=layer_name, inplace=True)

        return sd_perm

    if blocks is None:
        if model_type == 'llama-3.2-1b':
            blocks = LLAMA3_2_1B_ALL_BLOCKS

    sd_perm = permute_llama2_all_layers(sd, blocks=blocks, inplace=inplace)
    return sd_perm

llama_3_2_1b = partial(llama, model_type='llama-3.2-1b')

perm_map = {
    'vgg11': vgg11,
    'vgg16': vgg16,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'densenet121': densenet121,

    'llama-3.2-1b': llama_3_2_1b,
}

def permute_model(model_name:str, sd:OrderedDict, inplace:bool=True, **kwargs):
    return perm_map[model_name](sd, inplace=inplace, **kwargs)