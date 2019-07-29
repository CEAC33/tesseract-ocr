# tesseract-ocr

## Installing Tesseract for OCR

### Step #1: Install Tesseract
In order to use the Tesseract library, we first need to install it on our system.

For macOS users, we’ll be using Homebrew to install Tesseract:
```
$ brew install tesseract
```

If you’re using the Ubuntu operating system, simply use `apt-get` to install Tesseract OCR:
```
$ sudo apt-get install tesseract-ocr
```

### Step #2: Validate that Tesseract has been installed
To validate that Tesseract has been successfully installed on your machine, execute the following command:
```
$ tesseract -v
tesseract 4.0.0
 leptonica-1.78.0
  libgif 5.1.4 : libjpeg 9c : libpng 1.6.37 : libtiff 4.0.10 : zlib 1.2.11 : libwebp 1.0.3 : libopenjp2 2.3.1
 Found AVX2
 Found AVX
 Found SSE
```

### Step #3: Test out Tesseract OCR
For Tesseract OCR to obtain reasonable results, you’ll want to supply images that are cleanly pre-processed.

When utilizing Tesseract, I recommend:

- Using as an input image with as high resolution and DPI as possible.
- Applying thresholding to segment the text from the background.
- Ensuring the foreground is as clearly segmented from the background as possible (i.e., no pixelations or character deformations).
- Applying text skew correction to the input image to ensure the text is properly aligned.

Deviations from these recommendations can lead to incorrect OCR results as we’ll find out later in this tutorial.

Now, let’s apply OCR to the following image:

![alt text](https://www.pyimagesearch.com/wp-content/uploads/2017/05/example_01.png)

Simply enter the following command in your terminal:
```
$ tesseract tesseract_inputs/example_01.png stdout
Warning: Invalid resolution 0 dpi. Using 70 instead.
Estimating resolution as 241
Testing Tesseract OCR
```

Correct! Tesseract correctly identified, “Testing Tesseract OCR”, and printed it in the terminal.

Next, let’s try this image:

![alt text](https://www.pyimagesearch.com/wp-content/uploads/2017/05/example_02.png)

Enter the following in your terminal, noting the changed input filename:
```
$ tesseract tesseract_inputs/example_02.png stdout
Warning: Invalid resolution 0 dpi. Using 70 instead.
Estimating resolution as 331
PyImageSearch
```

Success! Tesseract correctly identified the text, “PyImageSearch”, in the image.

Now, let’s try OCR’ing digits as opposed to alphabetic characters:

![alt text](https://www.pyimagesearch.com/wp-content/uploads/2017/05/example_03.png)

This example uses the command line  digits  switch to only report digits:

```
$ tesseract tesseract_inputs/example_03.png stdout digits
Warning: Invalid resolution 0 dpi. Using 70 instead.
Estimating resolution as 632
650 3428
```

Once again, Tesseract correctly identified our string of characters (in this case digits only).

In each of these three situations Tesseract was able to correctly OCR all of our images — and you may even be thinking that Tesseract is the right tool for all OCR uses cases.

However, as we’ll find out in the next section, Tesseract has a number of limitations.

### Limitations of Tesseract for OCR

A few weeks ago I was working on a project to recognize the 16-digit numbers on credit cards.

I was easily able to write Python code to localize each of the four groups of 4-digits.

Here is an example 4-digit region of interest:

![alt text](https://www.pyimagesearch.com/wp-content/uploads/2017/05/example_04.png)

However, when I tried to apply Tesseract to the following image, the results were dissatisfying:

```
$ tesseract tesseract_inputs/example_04.png stdout digits
Warning: Invalid resolution 0 dpi. Using 70 instead.
Estimating resolution as 197
Sb1B
```

Notice how Tesseract reported `Sb1B` , but the image clearly shows 5678 .
Unfortunately, this is a great example of a limitation of Tesseract. While we have segmented the foreground text from background, the pixelated nature of the text “confuses” Tesseract. It’s also likely that Tesseract was not trained on a credit card-like font.

Tesseract is best suited when building document processing pipelines where images are scanned in, pre-processed, and then Optical Character Recognition needs to be applied.

We should note that Tesseract is not an off-the-shelf solution to OCR that will work in all (or even most) image processing and computer vision applications.

In order to accomplish that, you’ll need to apply feature extraction techniques, machine learning, and deep learning.

### Summary

We learned how to install and configure Tesseract on our machines, the first part in a two part series on using Tesseract for OCR. We then used the  tesseract  binary to apply OCR to input images.

However, we found out that unless our images are cleanly segmented Tesseract will give poor results. In the case of “noisy” input images, we’ll likely obtain better accuracy by training a custom machine learning model to recognize characters in our specific use case.

Tesseract is best suited for situations with high resolution inputs where the foreground text is cleanly segmented from the background.

## Using Tesseract OCR with Python

### Installing the Tesseract + Python “bindings”

```bash
workon cv
```

```bash
pip install pillow
pip install pytesseract
```

**Note:** pytesseract  does not provide true Python bindings. Rather, it simply provides an interface to the tesseract  binary. If you take a look at the project on GitHub you’ll see that the library is writing the image to a temporary file on disk followed by calling the tesseract  binary on the file and capturing the resulting output. This is definitely a bit hackish, but it gets the job done for us.

### Applying OCR with Tesseract and Python

Let’s begin by creating a new file named `ocr.py`:
```python
# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
	help="type of preprocessing to be done")
args = vars(ap.parse_args())
```

**Lines 2-6** handle our imports. The Image  class is required so that we can load our input image from disk in PIL format, a requirement when using pytesseract .

Our command line arguments are parsed on Lines 9-14. We have two command line arguments:

* `--image`: The path to the image we’re sending through the OCR system.
* `--preprocess`: The preprocessing method. This switch is optional and for this tutorial and can accept two values:  thresh  (threshold) or blur .

Next we’ll load the image, binarize it, and write it to disk.

```python
# load the example image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# check to see if we should apply thresholding to preprocess the
# image
if args["preprocess"] == "thresh":
	gray = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
 
# make a check to see if median blurring should be done to remove
# noise
elif args["preprocess"] == "blur":
	gray = cv2.medianBlur(gray, 3)
 
# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)
```

First, we load `--image` from disk into memory (**Line 17**) followed by converting it to grayscale (**Line 18**).

Next, depending on the pre-processing method specified by our command line argument, we will either threshold or blur the image. This is where you would want to add more advanced pre-processing methods (depending on your specific application of OCR) which are beyond the scope of this blog post.

The `if` statement and body on **Lines 22-24** perform a threshold in order to segment the foreground from the background. We do this using both  `cv2.THRESH_BINARY` and `cv2.THRESH_OTSU` flags. For details on Otsu’s method, see “Otsu’s Binarization” in the official OpenCV documentation. (https://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html)

We will see later in the results section that this thresholding method can be useful to read dark text that is overlaid upon gray shapes.

Alternatively, a blurring method may be applied. **Lines 28-29** perform a median blur when the `--preprocess` flag is set to `blur` . Applying a median blur can help reduce salt and pepper noise, again making it easier for Tesseract to correctly OCR the image.

After pre-processing the image, we use `os.getpid` to derive a temporary image filename based on the process ID of our Python script (**Line 33**).

The final step before using `pytesseract` for OCR is to write the pre-processed image, `gray`, to disk saving it with the `filename` from above (Line 34).

We can finally apply OCR to our image using the Tesseract Python “bindings”:
```python
# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)
 
# show the output images
cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)
```

Using `pytesseract.image_to_string` on **Line 38** we convert the contents of the image into our desired string, text . Notice that we passed a reference to the temporary image file residing on disk.

This is followed by some cleanup on **Line 39** where we delete the temporary file.

**Line 40** is where we print text to the terminal. In your own applications, you may wish to do some additional processing here such as spellchecking for OCR errors or Natural Language Processing rather than simply printing it to the console as we’ve done in this tutorial.

Finally, **Lines 43 and 44** handle displaying the original image and pre-processed image on the screen in separate windows. The `cv2.waitKey(0)` on Line 34 indicates that we should wait until a key on the keyboard is pressed before exiting the script.

### Tesseract OCR and Python results

Now that `ocr.py` has been created, it’s time to apply Python + Tesseract to perform OCR on some example input images.

In this section we will try OCR’ing three sample images using the following process:

- First, we will run each image through the Tesseract binary as-is.
- Then we will run each image through `ocr.py` (which performs pre-processing before sending through Tesseract).
- Finally, we will compare the results of both of these methods and note any errors.

Our first example is a “noisy” image. This image contains our desired foreground black text on a background that is partly white and partly scattered with artificially generated circular blobs. The blobs act as “distractors” to our simple algorithm.

![alt text](https://www.pyimagesearch.com/wp-content/uploads/2017/06/example_01.png)

Using the Tesseract binary, as we learned last week, we can apply OCR to the raw, unprocessed image:
```bash
$ tesseract images/example_01.png stdout
Warning: Invalid resolution 0 dpi. Using 70 instead.
Estimating resolution as 575
Noisyimage
to test
Tesseract OCR
```

Tesseract performed well with no errors in this case.

Now let’s confirm that our newly made script, ocr.py , also works:
```bash
$ python ocr.py --image images/example_01.png
Noisy image
to test
Tesseract OCR
```

![alt text](https://user-images.githubusercontent.com/51218415/62078448-4d724280-b212-11e9-877e-ec034bf24bf2.png)

As you can see in this screenshot, the thresholded image is very clear and the background has been removed. Our script correctly prints the contents of the image to the console.

Next, let’s test Tesseract and our pre-processing script on an image with “salt and pepper” noise in the background:

![alt text](https://www.pyimagesearch.com/wp-content/uploads/2017/06/example_02.png)

We can see the output of the tesseract  binary below:
```bash
$ tesseract images/example_02.png stdout
Warning: Invalid resolution 0 dpi. Using 70 instead.
Estimating resolution as 614
Detected 32 diacritics
_ Tesseract Will
Fail With Noisy
_ Backgrounds —
```

Unfortunately, Tesseract did not successfully OCR the text in the image.

However, by using the blur  pre-processing method in ocr.py  we can obtain better results:
```bash
$ python ocr.py --image images/example_02.png --preprocess blur
Tesseract Will
Fail With Noisy
Backgrounds
```

![alt text](https://user-images.githubusercontent.com/51218415/62079606-aba02500-b214-11e9-9aa2-27fdd796faee.png)

Success! Our blur pre-processing step enabled Tesseract to correctly OCR and output our desired text.

Finally, let’s try another image, this one with more text:

![alt text](https://www.pyimagesearch.com/wp-content/uploads/2017/06/example_03.png)

```bash
$ tesseract images/example_03.png stdout
Warning: Invalid resolution 0 dpi. Using 70 instead.
Estimating resolution as 157
PREREQUISITES

In order to make the most of this, you will need to have
alittle bit of programming experience. All examples in this
book are in the Python programming language. Familiarity
with Python or other scripting languages is suggested, but
not required.

You'll also need to know some basic mathematics. This
book is hands-on and example driven: lots of examples and
lots of code, so even if your math skills are not up to par,
do not worry! The examples are very detailed and heavily
documented to help you follow along.
```

Followed by testing the image with `ocr.py` :
```bash
$ python ocr.py --image images/example_03.png
PREREQUISITES

In order to make the most of this, you will need to have
alittle bit of programming experience. All examples in this
book are in the Python programming language. Familiarity
with Python or other scripting languages is suggested, but
not required.

You'll also need to know some basic mathematics. This
book is hands-on and example driven: lots of examples and
lots of code, so even if your math skills are not up to par,
do not worry! The examples are very detailed and heavily
documented to help you follow along.
```

Python + Tesseract did a reasonable job here, but once again we have demonstrated the limitations of the library as an off-the-shelf classifier.

We may obtain good or acceptable results with Tesseract for OCR, but the best accuracy will come from training custom character classifiers on specific sets of fonts that appear in actual real-world images.

Don’t let the results of Tesseract OCR discourage you — simply manage your expectations and be realistic on Tesseract’s performance. There is no such thing as a true “off-the-shelf” OCR system that will give you perfect results (there are bound to be some errors).

**Note:** If your text is rotated, you may wish to do additional pre-processing. Otherwise, if you’re interested in building a mobile document scanner, you now have a reasonably good OCR system to integrate into it.

### Summary

We learned how to apply the Tesseract OCR engine with the Python programming language. This enabled us to apply OCR algorithms from within our Python script.

The biggest downside is with the limitations of Tesseract itself. Tesseract works best when there are extremely clean segmentations of the foreground text from the background.

Furthermore these segmentations need to be as high resolution (DPI) as possible and the characters in the input image cannot appear “pixelated” after segmentation. If characters do appear pixelated then Tesseract will struggle to correctly recognize the text — we found this out even when applying images captured under ideal conditions (a PDF screenshot).

OCR, while no longer a new technology, is still an active area of research in the computer vision literature especially when applying OCR to real-world, unconstrained images. Deep learning and Convolutional Neural Networks (CNNs) are certainly enabling us to obtain higher accuracy, but we are still a long way from seeing “near perfect” OCR systems. Furthermore, as OCR has many applications across many domains, some of the best algorithms used for OCR are commercial and require licensing to be used in your own projects.

My primary suggestion to readers when applying OCR to their own projects is to first try Tesseract and if results are undesirable move on to the Google Vision API. (https://cloud.google.com/vision/)

*If neither **Tesseract** nor the **Google Vision API** obtain reasonable accuracy*, you might want to reassess your dataset and decide if it’s worth it to train your own custom character classifier — this is especially true if your dataset is noisy and/or contains very specific fonts you wish to detect and recognize. Examples of specific fonts include the digits on a credit card, the account and routing numbers found at the bottom of checks, or stylized text used in graphic design.
