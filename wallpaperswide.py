# https://wallpaperswide.com/games-desktop-wallpapers.html
# https://wallpaperswide.com/spider_man_miles_morales_art-wallpapers.html

# import re
# from bs4 import BeautifulSoup
# from numba import jit
from typing import Union, override
from urllib.request import Request, urlopen

WALLPAPERSWIDE_BASE_URL: str = r"https://wallpaperswide.com"
WALLPAPERSWIDE_WALLPAPER_DOWNLOAD_TEMPLATE_URL: str = r"https://wallpaperswide.com/download/{}"


class WallpaperCategory(object):
    WALLPAPERSWIDE_WALLPAPER_CATEGORY_TEMPLATE_URL: str = r"https://wallpaperswide.com/{}-desktop-wallpapers.html"
    WALLPAPERSWIDE_VALID_WALLPAPER_CATEGORIES: list[str] = [
        "aero",
        "animals",
        "architecture",
        "army",
        "artistic",
        "awareness",
        "black_and_white",
        "cartoons",
        "celebrities",
        "city",
        "computers",
        "cute",
        "elements",
        "food_and_drink",
        "funny",
        "games",
        "girls",
        "holidays",
        "love",
        "motors",
        "movies",
        "music",
        "nature",
        "seasons",
        "space",
        "sports",
        "travel",
        "vintage",
    ]

    def __init__(self, _category: str) -> None:
        if _category not in WallpaperCategory.WALLPAPERSWIDE_VALID_WALLPAPER_CATEGORIES:
            raise ValueError(
                f"Invalid wallpaper category {_category}, expected on of {WallpaperCategory.WALLPAPERSWIDE_VALID_WALLPAPER_CATEGORIES}"
            )
        self._category: str = _category
        self._url: str = WallpaperCategory.WALLPAPERSWIDE_WALLPAPER_CATEGORY_TEMPLATE_URL.format(_category)

    @override
    def __repr__(self) -> str:
        return f"{self._category} :: url <<{self._url}>>"

    def url(self) -> str:
        return self._url


class FirefoxImpersonator(Request):
    def __init__(self, url: str) -> None:
        super().__init__(
            url,
            data=None,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:138.0) Gecko/20100101 Firefox/138.0"},
            origin_req_host=None,
            unverifiable=False,
            method="GET",
        )


def download_category_firstpage_html(category: WallpaperCategory) -> str | None:
    try:
        with urlopen(FirefoxImpersonator(category.url())) as connection:
            html_page = connection.read()
    except BaseException as error:
        raise RuntimeError(f"{error.__dict__}") from error
    return html_page


def harvest_wallpaper_webpage_html(_wallpaper_url: str) -> str:
    pass


def extract_best_169_resolution_link(wallpaper_resolutions_html_div: str) -> Union[str, None]:
    """
    16:9
    """
    pass


def extract_best_1610_resolution_link(wallpaper_resolutions_html_div: str) -> Union[str, None]:
    """
    16:10
    """
    pass


def main() -> None:
    first_page: str | None = download_category_firstpage_html(WallpaperCategory(r"girls"))
    print(first_page)


if __name__ == "__main__":
    main()
