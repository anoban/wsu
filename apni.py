def parse_apni_name_list(path_to_html: str | None = None) -> "NDArray[str]":
    """
    Extracts and returns the species list from Australian Plant Name Index (APNI) HTML file,
    found through the hyperlink "Total plant name list for Australia (7,203 KB)" @ https://www.anbg.gov.au/apni/apni.html

    Required external modules:: bs4, Numpy
    Required std modules:: re

    Returns:: NDArray[str]
    """

    APNI_TOTAL_PLANT_NAME_LIST_URI: str = "https://www.anbg.gov.au/apni/apni.html"

    try:
        # standard library imports
        import re
        from urllib.request import urlopen

        PATTERN = re.compile(r"www.anbg.gov.au/cgi-bin/apx\?taxon_id=[\d]+", re.IGNORECASE)

        # third party module imports
        import numpy as np
        from bs4 import BeautifulSoup

        if path_to_html:  # if the parameter is None
            with open(file=path_to_html, mode="r") as fp:
                soup = BeautifulSoup(markup=fp.read(), features=r"html.parser")
        else:
            with urlopen(url=APNI_TOTAL_PLANT_NAME_LIST_URI) as connexion:
                soup = BeautifulSoup(markup=connexion.read(), features=r"html.parser")

    except ModuleNotFoundError as mnf_err:
        raise RuntimeError("This function requires modules numpy and bs4 to be installed on the machine!") from mnf_err
    except FileNotFoundError as fnf_err:
        raise RuntimeError(f"Invalid path! {path_to_html}") from fnf_err

    return np.array([_.text for _ in soup.find_all("a", attrs={"href": PATTERN})])
