from .kgw_detector import KGWDetector
from .kth_detector import KTHDetector
from .aar_detector import AARDetector
from .dip_detector import DIPDetector
from .its_detector import ITSDetector
from .waterbag_detector import WaterbagDetector

DETECTOR_MAP = {
    "kgw": KGWDetector,
    "kth": KTHDetector,
    "aar": AARDetector,
    "dip": DIPDetector,
    "its": ITSDetector,
    "waterbag": WaterbagDetector,
}
