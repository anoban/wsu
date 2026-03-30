___Stripped down segmentation models___
------------------------------------------
	1. SAM (Segment Anything Model)
	2. SAM 2.1
	3. MaskRCNN
--------------------------------


___ImageJ___
------------------------------

- Downlaod and install [JRE 8u481](https://www.java.com/en/download/), newer Java versions won't (didn't) work. `ImageJ` requires `Java 8`.
- Download and extract the `ImageJ` [archive](https://imagej.net/ij/download.html).
- Create or edit (if already exists) the `ImageJ.cfg` file, as follows:

    ```
    .
    <path to javaw.exe>
    -Xmx3096m -cp ij.jar ij.ImageJ
    ```

    e.g. path to `javaw.exe` can be `C:\Program Files\Java\jre1.8.0_481\bin\javaw.exe`
    and the `-Xmx` argument specifies maximum memory use, which is in this example set to `3096MiBs`.
