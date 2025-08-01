import re
from urllib.request import urlopen

import numpy as np
from bs4 import BeautifulSoup
from numpy.typing import NDArray

__all__ = ("parse_apni_name_list",)


def parse_apni_name_list(path_to_html: str | None = None) -> NDArray[np.str_]:
    """
    Extracts and returns the species list from Australian Plant Name Index (APNI) HTML file,
    found through the hyperlink "Total plant name list for Australia (7,203 KB)" @ https://www.anbg.gov.au/apni/apni.html

    Required external modules:: bs4, Numpy
    Required std modules:: re

    Returns:: NDArray[np.str_]
    """

    APNI_TOTAL_PLANT_NAME_LIST_URI: str = "https://www.anbg.gov.au/apni/apni.html"

    PATTERN = re.compile(r"www.anbg.gov.au/cgi-bin/apx\?taxon_id=[\d]+", re.IGNORECASE)

    if path_to_html:  # if the parameter is not None
        try:
            # working with a local file seems terribly slow comapred to a file downloaded on the fly from internet????
            with open(file=path_to_html, mode="r") as fp:
                soup = BeautifulSoup(markup=fp.read(), features=r"html.parser")
        except FileNotFoundError as fnf_err:
            raise RuntimeError(f"Invalid path! {path_to_html}") from fnf_err
    else:
        with urlopen(url=APNI_TOTAL_PLANT_NAME_LIST_URI) as connexion:
            soup = BeautifulSoup(markup=connexion.read(), features=r"html.parser")

    return np.array([_.text for _ in soup.find_all("a", attrs={"href": PATTERN})], dtype=str)
