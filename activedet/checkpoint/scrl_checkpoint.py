import copy
import re

import torch
from torchvision import models

from .torchvision_checkpoint import torchvision_to_d2, convert_torchvision_names

def generate_key_map():
    """We wish to generate map that will directly convert the Kakao keys into the original torchvision.
    The main issue is the fact they used nn.Sequential which removed the names of the original torchvision children
    """
    network = models.resnet50(pretrained=False)
    # This is how Kakaobrain initialized Resnet50 for SCRL
    encoder = torch.nn.Sequential(*list(network.children())[:-1])
    return {src:target for src,target in zip(encoder.state_dict(), network.state_dict())}


def convert_kakao_names(original_keys):
    layer_keys = copy.deepcopy(original_keys)
    # Get only the online network
    eval_keys = layer_keys['evaluator_state_dict']
    layer_keys = layer_keys['online_network_state_dict']
    
    # Remove all keys that are not encoder
    encoder_keys = {key:value for key,value in layer_keys.items() if 'encoder.' in key}
    encoder_keys = {re.sub("^encoder.","",key): value for key, value in encoder_keys.items()}
    
    key_map = generate_key_map()
    torchvision_keys = {key_map[kakao_key]: value for kakao_key, value in encoder_keys.items()}
    
    # Rename torchvision into detectron2 keys
    d2_keys = convert_torchvision_names(torchvision_keys)
    # d2_keys = {re.sub("^conv1\\.weight$","stem.conv1.weight",key): value for key, value in d2_keys.items()}    
    # d2_keys = {re.sub("^bn1\\.","stem.conv1.norm.",key): value for key, value in d2_keys.items()}
    # d2_keys = {re.sub("\\.bn1\\.",".conv1.norm.",key): value for key, value in d2_keys.items()}
    # d2_keys = {re.sub("\\.bn2\\.",".conv2.norm.",key): value for key, value in d2_keys.items()}
    # d2_keys = {re.sub("\\.bn3\\.",".conv3.norm.",key): value for key, value in d2_keys.items()}
    # d2_keys = {re.sub("^layer1.","res2.",key): value for key, value in d2_keys.items()}
    # d2_keys = {re.sub("^layer2.","res3.",key): value for key, value in d2_keys.items()}
    # d2_keys = {re.sub("^layer3.","res4.",key): value for key, value in d2_keys.items()}
    # d2_keys = {re.sub("^layer4.","res5.",key): value for key, value in d2_keys.items()}


    # Collect now the evaluator keys
#     evaluator_keys = {re.sub("^linear.","fc.",key): value for key,value in eval_keys.items()}
    d2_keys = {**d2_keys, **eval_keys}
    return d2_keys
    