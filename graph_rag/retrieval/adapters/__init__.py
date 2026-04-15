from .agris import AgrisAdapter
from .faostat import FaostatAdapter
from .cgiar import CgiarAdapter
from .climate import ClimateAdapter
from .soil import SoilAdapter
from .agricola import AgricolaAdapter
from .pubag import PubAgAdapter
from .cabi import CabiAdapter
from .agecon import AgEconAdapter
from .asabe import AsabeAdapter


def default_adapters():
    return [
        AgrisAdapter(),
        FaostatAdapter(),
        CgiarAdapter(),
        ClimateAdapter(),
        SoilAdapter(),
        AgricolaAdapter(),
        PubAgAdapter(),
        CabiAdapter(),
        AgEconAdapter(),
        AsabeAdapter(),
    ]
