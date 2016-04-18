# ChessDetect
Android App for finding a chessboard in a scene live and trying to predict what the piece layout is.

[![Android chessboard pre-detection stage](http://img.youtube.com/vi/2bs1eozPmVg/0.jpg)](https://youtu.be/2bs1eozPmVg)

* 4/17/2016 - Android app with pre-detection of chessboard bounding box + hough lines within the area.
* TODO: Prune to relevant chessboard lines only
* TODO: 

Uses OpenCV for Android, in addition to downloading this repo, 
you'll need to download and install [OpenCV for Android](http://docs.opencv.org/2.4/doc/tutorials/introduction/android_binary_package/O4A_SDK.html#manual-opencv4android-sdk-setup).
This was done to avoid reproducing the opencv library and jnis in this repo.

Then once you've opened this project in Android Studio, add the OpenCVLibrary310 module to this app.
Similarly, you'll need to copy over the native libs over to a new folder called `jniLibs`, as explained by [these instructions on StackOverflow](http://stackoverflow.com/a/27421494/2574639) were the ones that worked best for me

## Ideation

A little thought into the actual prediction step, it's likely unrealistic to assume computer vision can detect 
what pieces are on the board (so many chessboards with so many different model styles, not even counting 3D being hard).

Instead, given knowledge of the tile squares (which we will affine warp into tiles, say 32x32 like in tensorflow_chessbot), 
it should be doable to tell if a tile is empty, or contains a black piece, or a white piece. 

Then, it may be possible to train a tensorflow machine learning model where the input is 

* 64 length integer array, 0 = empty, 1 - white piece, 2 - black piece

and the output is the correct FEN notation.

We can generate extremely large datasets of this format from existing datasets of games played, where we have the correct output already,
in fact, every game with every move is a new sample, the output is provided, which we can convert to the input.

It remains to be seen whether given such a dataset and expectations of inputs to outputs if there is enough information to predict this successfully.
