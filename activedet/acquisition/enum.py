from enum import IntEnum, unique

@unique
class LabelCenterMode(IntEnum):
    """An enum of properties of a data in Core-Set"""

    UNLABELLED = 0
    """When the data is not yet labelled
    """

    CENTER = 1
    """When the data is considered as center
    """

    FOR_LABEL = 2
    """When the data is considered to be labelled during acquisition
    """
