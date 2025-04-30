# https://wallpaperswide.com/games-desktop-wallpapers.html
# https://wallpaperswide.com/spider_man_miles_morales_art-wallpapers.html

# import re
# from bs4 import BeautifulSoup
# from urllib.request import (urlopen, Request)
# from numba import jit

BASE_URL_WALLPAPERSWIDE: str = r"https://wallpaperswide.com/"


def extract_best_resolution(html_chunk: str) -> tuple[int, int]:
    """ """
    pass


# <div class="wallpaper-resolutions" id="wallpaper-resolutions">

"""
<h3>UHD 16:9</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-3840x2160.jpg" title="Download Spider Man Miles Morales Art 16:9 3840 x 2160 4K UHD Wallpaper for 2160p Ultra HD Desktop &amp; TV display">3840x2160</a>

                                <br clear="all">				<h3>HD 16:9</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-960x540.jpg" title="Download Spider Man Miles Morales Art 16:9 960 x 540 Wallpaper">960x540</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1024x576.jpg" title="Download Spider Man Miles Morales Art 16:9 1024 x 576 Wallpaper">1024x576</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1280x720.jpg" title="Download Spider Man Miles Morales Art 16:9 1280 x 720 HD Wallpaper for 720p High Definition Desktop &amp; TV display">1280x720</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1366x768.jpg" title="Download Spider Man Miles Morales Art 16:9 1366 x 768 HD Wallpaper">1366x768</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1600x900.jpg" title="Download Spider Man Miles Morales Art 16:9 1600 x 900 HD Wallpaper">1600x900</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1920x1080.jpg" title="Download Spider Man Miles Morales Art 16:9 1920 x 1080 HD Wallpaper for 1080p FHD Full HD Desktop &amp; TV display">1920x1080</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2048x1152.jpg" title="Download Spider Man Miles Morales Art 16:9 2048 x 1152 HD Wallpaper">2048x1152</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2400x1350.jpg" title="Download Spider Man Miles Morales Art 16:9 2400 x 1350 HD Wallpaper">2400x1350</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2560x1440.jpg" title="Download Spider Man Miles Morales Art 16:9 2560 x 1440 HD Wallpaper for 1440p QHD Quad HD Desktop &amp; TV display">2560x1440</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2880x1620.jpg" title="Download Spider Man Miles Morales Art 16:9 2880 x 1620 HD Wallpaper">2880x1620</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-3554x1999.jpg" title="Download Spider Man Miles Morales Art 16:9 3554 x 1999 HD Wallpaper">3554x1999</a>

                                <br clear="all">				<h3>UltraWide 21:9</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2560x1080.jpg" title="Download Spider Man Miles Morales Art  21:9 2560 x 1080 UltraWide HD Wallpaper">2560x1080</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-3440x1440.jpg" title="Download Spider Man Miles Morales Art  21:9 3440 x 1440 UltraWide HD Wallpaper">3440x1440</a>

                                <br clear="all">				<h3>UltraWide 24:10</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2880x1200.jpg" title="Download Spider Man Miles Morales Art  24:10 2880 x 1200 UltraWide HD Wallpaper">2880x1200</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-3840x1600.jpg" title="Download Spider Man Miles Morales Art  24:10 3840 x 1600 UltraWide 4K UHD Wallpaper">3840x1600</a>

                                <br clear="all">				<h3>UltraWide 32:9</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-3840x1080.jpg" title="Download Spider Man Miles Morales Art  32:9 3840 x 1080 UltraWide 4K UHD Wallpaper">3840x1080</a>

                                <br clear="all">				<h3>UltraWide 32:10</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2880x900.jpg" title="Download Spider Man Miles Morales Art  32:10 2880 x 900 UltraWide HD Wallpaper">2880x900</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-3840x1200.jpg" title="Download Spider Man Miles Morales Art  32:10 3840 x 1200 UltraWide 4K UHD Wallpaper">3840x1200</a>

                                <br clear="all">				<h3>Widescreen 16:10</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-960x600.jpg" title="Download Spider Man Miles Morales Art 16:10 960 x 600 Widescreen Wallpaper">960x600</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1152x720.jpg" title="Download Spider Man Miles Morales Art 16:10 1152 x 720 Widescreen Wallpaper">1152x720</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1280x800.jpg" title="Download Spider Man Miles Morales Art 16:10 1280 x 800 Widescreen HD Wallpaper">1280x800</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1440x900.jpg" title="Download Spider Man Miles Morales Art 16:10 1440 x 900 Widescreen HD Wallpaper">1440x900</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1680x1050.jpg" title="Download Spider Man Miles Morales Art 16:10 1680 x 1050 Widescreen HD Wallpaper">1680x1050</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1920x1200.jpg" title="Download Spider Man Miles Morales Art 16:10 1920 x 1200 Widescreen HD Wallpaper">1920x1200</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2560x1600.jpg" title="Download Spider Man Miles Morales Art 16:10 2560 x 1600 Widescreen HD Wallpaper">2560x1600</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2880x1800.jpg" title="Download Spider Man Miles Morales Art 16:10 2880 x 1800 Widescreen HD Wallpaper">2880x1800</a>

                                <br clear="all">				<h3>Widescreen 5:3</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-800x480.jpg" title="Download Spider Man Miles Morales Art 5:3 800 x 480 Widescreen Wallpaper">800x480</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1280x768.jpg" title="Download Spider Man Miles Morales Art 5:3 1280 x 768 Widescreen HD Wallpaper">1280x768</a>

                                <br clear="all">				<h3>Fullscreen 4:3</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-800x600.jpg" title="Download Spider Man Miles Morales Art 4:3 800 x 600 Fullscreen Wallpaper">800x600</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1024x768.jpg" title="Download Spider Man Miles Morales Art 4:3 1024 x 768 Fullscreen Wallpaper">1024x768</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1152x864.jpg" title="Download Spider Man Miles Morales Art 4:3 1152 x 864 Fullscreen Wallpaper">1152x864</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1280x960.jpg" title="Download Spider Man Miles Morales Art 4:3 1280 x 960 Fullscreen HD Wallpaper">1280x960</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1400x1050.jpg" title="Download Spider Man Miles Morales Art 4:3 1400 x 1050 Fullscreen HD Wallpaper">1400x1050</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1440x1080.jpg" title="Download Spider Man Miles Morales Art 4:3 1440 x 1080 Fullscreen HD Wallpaper">1440x1080</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1600x1200.jpg" title="Download Spider Man Miles Morales Art 4:3 1600 x 1200 Fullscreen HD Wallpaper">1600x1200</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1680x1260.jpg" title="Download Spider Man Miles Morales Art 4:3 1680 x 1260 Fullscreen HD Wallpaper">1680x1260</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1920x1440.jpg" title="Download Spider Man Miles Morales Art 4:3 1920 x 1440 Fullscreen HD Wallpaper">1920x1440</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2048x1536.jpg" title="Download Spider Man Miles Morales Art 4:3 2048 x 1536 Fullscreen HD Wallpaper">2048x1536</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2560x1920.jpg" title="Download Spider Man Miles Morales Art 4:3 2560 x 1920 Fullscreen HD Wallpaper">2560x1920</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2800x2100.jpg" title="Download Spider Man Miles Morales Art 4:3 2800 x 2100 Fullscreen HD Wallpaper">2800x2100</a>

                                <br clear="all">				<h3>Fullscreen 5:4</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1280x1024.jpg" title="Download Spider Man Miles Morales Art 5:4 1280 x 1024 Fullscreen HD Wallpaper">1280x1024</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1920x1536.jpg" title="Download Spider Man Miles Morales Art 5:4 1920 x 1536 Fullscreen HD Wallpaper">1920x1536</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2560x2048.jpg" title="Download Spider Man Miles Morales Art 5:4 2560 x 2048 Fullscreen HD Wallpaper">2560x2048</a>

                                <br clear="all">				<h3>Fullscreen 3:2</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-960x640.jpg" title="Download Spider Man Miles Morales Art 3:2 960 x 640 Fullscreen Wallpaper">960x640</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1152x768.jpg" title="Download Spider Man Miles Morales Art 3:2 1152 x 768 Fullscreen Wallpaper">1152x768</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1440x960.jpg" title="Download Spider Man Miles Morales Art 3:2 1440 x 960 Fullscreen HD Wallpaper">1440x960</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1920x1280.jpg" title="Download Spider Man Miles Morales Art 3:2 1920 x 1280 Fullscreen HD Wallpaper">1920x1280</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2000x1333.jpg" title="Download Spider Man Miles Morales Art 3:2 2000 x 1333 Fullscreen HD Wallpaper">2000x1333</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2160x1440.jpg" title="Download Spider Man Miles Morales Art 3:2 2160 x 1440 Fullscreen HD Wallpaper">2160x1440</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2736x1824.jpg" title="Download Spider Man Miles Morales Art 3:2 2736 x 1824 Fullscreen HD Wallpaper">2736x1824</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2880x1920.jpg" title="Download Spider Man Miles Morales Art 3:2 2880 x 1920 Fullscreen HD Wallpaper">2880x1920</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-3000x2000.jpg" title="Download Spider Man Miles Morales Art 3:2 3000 x 2000 Fullscreen HD Wallpaper">3000x2000</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-3240x2160.jpg" title="Download Spider Man Miles Morales Art 3:2 3240 x 2160 Fullscreen HD Wallpaper">3240x2160</a>

                                <br clear="all">				<h3>Tablet 1:1</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1024x1024.jpg" title="Download Spider Man Miles Morales Art 1:1 1024 x 1024 Tablet Wallpaper for Apple iPad 1 &amp; 2, iPad Mini, Amazon Kindle Fire, etc .">1024x1024</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1280x1280.jpg" title="Download Spider Man Miles Morales Art 1:1 1280 x 1280 Tablet HD Wallpaper for most Android tablets like Samsung Galaxy Note &amp; Galaxy Tab series .">1280x1280</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2048x2048.jpg" title="Download Spider Man Miles Morales Art 1:1 2048 x 2048 Tablet HD Wallpaper for QXGA tablets such as any Apple iPad with Retina display, iPad 3 &amp; 4, iPad Air 1 &amp; 2, iPad Mini 2,3,4 and the 9.7-inch version of iPad Pro">2048x2048</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2160x2160.jpg" title="Download Spider Man Miles Morales Art 1:1 2160 x 2160 Tablet HD Wallpaper">2160x2160</a>

                                <br clear="all">				<h3>Mobile 9:16</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-360x640.jpg" title="Download Spider Man Miles Morales Art Mobile 9:16 360 x 640 Phone Wallpaper">360x640</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-480x854.jpg" title="Download Spider Man Miles Morales Art Mobile 9:16 480 x 854 Phone Wallpaper">480x854</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-540x960.jpg" title="Download Spider Man Miles Morales Art Mobile 9:16 540 x 960 Phone Wallpaper">540x960</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-720x1280.jpg" title="Download Spider Man Miles Morales Art Mobile 9:16 720 x 1280 Phone Wallpaper">720x1280</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1080x1920.jpg" title="Download Spider Man Miles Morales Art Mobile 9:16 1080 x 1920 Phone Wallpaper" style="background-color: rgb(255, 40, 0); color: rgb(255, 255, 255); font-weight: bold; border-radius: 6px; padding-left: 5px; padding-right: 5px;">1080x1920</a>

                                <br clear="all">				<h3>Mobile 9:19.5</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-720x1560.jpg" title="Download Spider Man Miles Morales Art Mobile 9:19.5 720 x 1560 Phone Wallpaper">720x1560</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-830x1800.jpg" title="Download Spider Man Miles Morales Art Mobile 9:19.5 830 x 1800 Phone Wallpaper">830x1800</a>

                                <br clear="all">				<h3>Mobile 9:20</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-720x1600.jpg" title="Download Spider Man Miles Morales Art Mobile 9:20 720 x 1600 Phone Wallpaper">720x1600</a>

                                <br clear="all">				<h3>Mobile 10:16</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-800x1280.jpg" title="Download Spider Man Miles Morales Art Mobile 10:16 800 x 1280 Phone Wallpaper">800x1280</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1200x1920.jpg" title="Download Spider Man Miles Morales Art Mobile 10:16 1200 x 1920 Phone Wallpaper">1200x1920</a>

                                <br clear="all">				<h3>Mobile 2:3</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-320x480.jpg" title="Download Spider Man Miles Morales Art Mobile 2:3 320 x 480 Phone Wallpaper for HVGA mobile devices e.g., Apple iPhone 1, 3G, 3GS, Apple iPod 1-3 .">320x480</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-640x960.jpg" title="Download Spider Man Miles Morales Art Mobile 2:3 640 x 960 Phone Wallpaper for DVGA or qHD mobile devices e.g., Apple iPhone 4/4S .">640x960</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-768x1152.jpg" title="Download Spider Man Miles Morales Art Mobile 2:3 768 x 1152 Phone Wallpaper for DVGA or qHD mobile devices e.g., Apple iPhone 4/4S .">768x1152</a>

                                <br clear="all">				<h3>Mobile 3:5</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-240x400.jpg" title="Download Spider Man Miles Morales Art Mobile 3:5 240 x 400 Phone Wallpaper for WQVGA mobile devices.">240x400</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-480x800.jpg" title="Download Spider Man Miles Morales Art Mobile 3:5 480 x 800 Phone Wallpaper for WVGA mobile devices.">480x800</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-768x1280.jpg" title="Download Spider Man Miles Morales Art Mobile 3:5 768 x 1280 Phone Wallpaper for WVGA mobile devices.">768x1280</a>

                                <br clear="all">				<h3>Mobile 3:4</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-240x320.jpg" title="Download Spider Man Miles Morales Art Mobile 3:4 240 x 320 Phone Wallpaper for QVGA mobile devices e.g., iPod 5, iPod 6, Zune">240x320</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-480x640.jpg" title="Download Spider Man Miles Morales Art Mobile 3:4 480 x 640 Phone Wallpaper for VGA mobile devices.">480x640</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-600x800.jpg" title="Download Spider Man Miles Morales Art Mobile 3:4 600 x 800 Phone Wallpaper for VGA mobile devices.">600x800</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-768x1024.jpg" title="Download Spider Man Miles Morales Art Mobile 3:4 768 x 1024 Phone Wallpaper for VGA mobile devices.">768x1024</a>

                                <br clear="all">				<h3>2X Widescreen 16:10</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1920x600.jpg" title="Download Spider Man Miles Morales Art 16:10 1920 x 600 Dual Wallpaper">1920x600</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2304x720.jpg" title="Download Spider Man Miles Morales Art 16:10 2304 x 720 Dual Wallpaper">2304x720</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2560x800.jpg" title="Download Spider Man Miles Morales Art 16:10 2560 x 800 Dual HD Wallpaper">2560x800</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2880x900.jpg" title="Download Spider Man Miles Morales Art 16:10 2880 x 900 Dual HD Wallpaper">2880x900</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-3360x1050.jpg" title="Download Spider Man Miles Morales Art 16:10 3360 x 1050 Dual HD Wallpaper">3360x1050</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-3840x1200.jpg" title="Download Spider Man Miles Morales Art 16:10 3840 x 1200 Dual HD Wallpaper">3840x1200</a>

                                <br clear="all">				<h3>2X Widescreen 5:3</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1600x480.jpg" title="Download Spider Man Miles Morales Art 5:3 1600 x 480 Dual Wallpaper">1600x480</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2560x768.jpg" title="Download Spider Man Miles Morales Art 5:3 2560 x 768 Dual HD Wallpaper">2560x768</a>

                                <br clear="all">				<h3>2X HD 16:9</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1920x540.jpg" title="Download Spider Man Miles Morales Art 16:9 1920 x 540 Dual Wallpaper">1920x540</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2048x576.jpg" title="Download Spider Man Miles Morales Art 16:9 2048 x 576 Dual Wallpaper">2048x576</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2560x720.jpg" title="Download Spider Man Miles Morales Art 16:9 2560 x 720 Dual HD Wallpaper">2560x720</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2732x768.jpg" title="Download Spider Man Miles Morales Art 16:9 2732 x 768 Dual HD Wallpaper">2732x768</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-3200x900.jpg" title="Download Spider Man Miles Morales Art 16:9 3200 x 900 Dual HD Wallpaper">3200x900</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-3840x1080.jpg" title="Download Spider Man Miles Morales Art 16:9 3840 x 1080 Dual HD Wallpaper">3840x1080</a>

                                <br clear="all">				<h3>2X Fullscreen 4:3</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-1600x600.jpg" title="Download Spider Man Miles Morales Art 4:3 1600 x 600 Dual Wallpaper">1600x600</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2048x768.jpg" title="Download Spider Man Miles Morales Art 4:3 2048 x 768 Dual Wallpaper">2048x768</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2304x864.jpg" title="Download Spider Man Miles Morales Art 4:3 2304 x 864 Dual Wallpaper">2304x864</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2560x960.jpg" title="Download Spider Man Miles Morales Art 4:3 2560 x 960 Dual HD Wallpaper">2560x960</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2800x1050.jpg" title="Download Spider Man Miles Morales Art 4:3 2800 x 1050 Dual HD Wallpaper">2800x1050</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2880x1080.jpg" title="Download Spider Man Miles Morales Art 4:3 2880 x 1080 Dual HD Wallpaper">2880x1080</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-3200x1200.jpg" title="Download Spider Man Miles Morales Art 4:3 3200 x 1200 Dual HD Wallpaper">3200x1200</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-3360x1260.jpg" title="Download Spider Man Miles Morales Art 4:3 3360 x 1260 Dual HD Wallpaper">3360x1260</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-3840x1440.jpg" title="Download Spider Man Miles Morales Art 4:3 3840 x 1440 Dual HD Wallpaper">3840x1440</a>

                                <br clear="all">				<h3>2X Fullscreen 5:4</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2560x1024.jpg" title="Download Spider Man Miles Morales Art 5:4 2560 x 1024 Dual HD Wallpaper">2560x1024</a>

                                <br clear="all">				<h3>2X Fullscreen 3:2</h3><em class="ui-icon ui-icon-triangle-1-e"></em>						<a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2304x768.jpg" title="Download Spider Man Miles Morales Art 3:2 2304 x 768 Dual Wallpaper">2304x768</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-2880x960.jpg" title="Download Spider Man Miles Morales Art 3:2 2880 x 960 Dual HD Wallpaper">2880x960</a>

                        <a target="_self" href="/download/spider_man_miles_morales_art-wallpaper-3840x1280.jpg" title="Download Spider Man Miles Morales Art 3:2 3840 x 1280 Dual HD Wallpaper">3840x1280</a>
<br clear="all">

"""

# </div>
