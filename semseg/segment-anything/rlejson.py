# convert the raw JSON annotation files created by labelme to RLE compressed JSON files, which will be convenient when training on servers
# we do not want to upload huge JSON files, do we???


class StrippedLabelmeAnnotation:
    def __init__(self, fpath: str) -> None:
        pass

    def shape(self) -> tuple[int, int]:
        pass

    def extract_shape_and_corrds():
        pass

    def runlength_encode():
        pass

    def to_coco_rle():
        pass
