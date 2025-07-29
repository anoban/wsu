import ssl
from urllib.request import Request, urlopen

# archive.org vidoes have crappy quality
# the course materials on youtube are 1080p :/

if __name__ == r"__main__":
    ARCHIVE_BASE_URI: str = r"https://ia800702.us.archive.org/1/items/MIT18.650F16/"
    LECTURE_VIDEO_TEMPLATE: str = r"MIT18_650F16_lec{:02d}_300k.mp4"

    for i in range(1, 25):
        opener = Request(ARCHIVE_BASE_URI + LECTURE_VIDEO_TEMPLATE.format(i))
        opener.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:137.0) Gecko/20100101 Firefox/137.0")
        with urlopen(opener, context=ssl.SSLContext()) as request:
            print(f"Downloading {LECTURE_VIDEO_TEMPLATE.format(i)}")
            video = request.read()
            with open(file=LECTURE_VIDEO_TEMPLATE.format(i), mode="wb") as fp:
                fp.write(video)
            print("Downloading completed!")
