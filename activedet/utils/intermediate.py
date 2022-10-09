

def get_attribute_recursive(obj, target_class,clues=['dataset','_dataset']):
    """We look for a certain object
    """
    attrs_list = dir(obj)
    candidate = [attr for attr in attrs_list if attr in clues]
    if not candidate:
        raise AttributeError("Cannot find {} instance in the object".format(target_class))
    # Let's assume one candidate first
    candidate = candidate[0]
    child = getattr(obj,candidate)
    if isinstance(child,target_class):
        return child
    return get_attribute_recursive(child,target_class,clues=clues)

