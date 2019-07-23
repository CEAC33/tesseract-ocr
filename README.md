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

Today we learned how to install and configure Tesseract on our machines, the first part in a two part series on using Tesseract for OCR. We then used the  tesseract  binary to apply OCR to input images.

However, we found out that unless our images are cleanly segmented Tesseract will give poor results. In the case of “noisy” input images, we’ll likely obtain better accuracy by training a custom machine learning model to recognize characters in our specific use case.

Tesseract is best suited for situations with high resolution inputs where the foreground text is cleanly segmented from the background.
