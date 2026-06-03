from .unwatermarked import UNWatermark
from .aar import AARWatermark
from .dip import DIPWatermark
from .kgw import KGWWatermark
from .kth import KTHWatermark
from .its import ITSWatermark
from .waterbag import WaterbagWatermark

WATERMARK_MAP = {
    "unwatermarked": UNWatermark,
    "aar": AARWatermark,
    "dip": DIPWatermark,
    "kgw": KGWWatermark,
    "kth": KTHWatermark,
    "its": ITSWatermark,
    "waterbag": WaterbagWatermark,
}
