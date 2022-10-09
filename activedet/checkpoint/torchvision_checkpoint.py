import copy
import re

def convert_torchvision_names(original_keys):
    torchvision_keys = copy.deepcopy(original_keys)

    # Rename torchvision into detectron2 keys
    d2_keys = torchvision_keys
    d2_keys = {re.sub("^conv1\\.weight$","stem.conv1.weight",key): value for key, value in d2_keys.items()}    #affect the first one only
    d2_keys = {re.sub("^bn1\\.","stem.conv1.norm.",key): value for key, value in d2_keys.items()} #affects the batch norms in the ste,
    d2_keys = {re.sub("\\.bn1\\.",".conv1.norm.",key): value for key, value in d2_keys.items()}
    d2_keys = {re.sub("\\.bn2\\.",".conv2.norm.",key): value for key, value in d2_keys.items()}
    d2_keys = {re.sub("\\.bn3\\.",".conv3.norm.",key): value for key, value in d2_keys.items()}
    d2_keys = {re.sub("^layer1.","res2.",key): value for key, value in d2_keys.items()}
    d2_keys = {re.sub("^layer2.","res3.",key): value for key, value in d2_keys.items()}
    d2_keys = {re.sub("^layer3.","res4.",key): value for key, value in d2_keys.items()}
    d2_keys = {re.sub("^layer4.","res5.",key): value for key, value in d2_keys.items()}
    d2_keys = {re.sub("downsample.0","shortcut", key): value for key, value in d2_keys.items()}
    d2_keys = {re.sub("downsample.1","shortcut.norm", key): value for key, value in d2_keys.items()}

    return d2_keys

def convert_torchvision_adobe(original_keys):
    torchvision_keys = copy.deepcopy(original_keys)

    # Rename torchvision into detectron2 keys
    d2_keys = torchvision_keys
    d2_keys = {re.sub("^conv1\\.weight$","stem.conv1.weight",key): value for key, value in d2_keys.items()}    #affect the first one only
    d2_keys = {re.sub("^bn1\\.","stem.conv1.norm.",key): value for key, value in d2_keys.items()} #affects the batch norms in the ste,
    d2_keys = {re.sub("\\.bn1\\.",".conv1.norm.",key): value for key, value in d2_keys.items()}
    d2_keys = {re.sub("\\.bn2\\.",".conv2.norm.",key): value for key, value in d2_keys.items()}
    d2_keys = {re.sub("\\.bn3\\.",".conv3.norm.",key): value for key, value in d2_keys.items()}
    d2_keys = {re.sub("^layer1.","res2.",key): value for key, value in d2_keys.items()}
    d2_keys = {re.sub("^layer2.","res3.",key): value for key, value in d2_keys.items()}
    d2_keys = {re.sub("^layer3.","res4.",key): value for key, value in d2_keys.items()}
    d2_keys = {re.sub("^layer4.","res5.",key): value for key, value in d2_keys.items()}

    new_key = {}
    for key, value in d2_keys.items():
        if "downsample"in key:
            # Check if key has .filt
            layer = ".".join(key.split('.')[:3])
            all_layers = [key for key in d2_keys.keys() if layer in key]
            if any(['.filt' in l for l in all_layers]) is True:
                key = re.sub("downsample.0","blur_shortcut", key)
                key = re.sub("downsample.1","shortcut", key)
                key = re.sub("downsample.2","shortcut.norm", key)
                new_key[key] = value
            else:
                key = re.sub("downsample.0","shortcut", key)
                key = re.sub("downsample.1","shortcut.norm", key)
                new_key[key]= value
        else:
            key = re.sub("maxpool.1","stem.blurpool",key)
            key = re.sub("^fc","linear", key)
            key = re.sub("conv3.0.filt","blur2.filt", key)
            key = re.sub("conv3.1.weight","conv3.weight",key)
            new_key[key] = value

    return new_key

def torchvision_to_d2(original_keys):
    """Mostly copy from detectron2
    """
    torchvision_keys = copy.deepcopy(original_keys)

    newmodel = {}
    for k in list(torchvision_keys.keys()):
        old_k = k
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = torchvision_keys.pop(old_k)
    
    return newmodel