package xyz.tetration.sam.chessdetect;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "ChessDetectActivity";
    private static final int REQUEST_CAMERA_PERMISSION = 1;
    private static final double THETA_THRESHOLD = 0.5 * Math.PI / 180;
    private static final double RHO_THRESHOLD = 1;
    private static final short HOUGH_ACCUMULATOR_THRESHOLD = 40;
    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.setMaxFrameSize(320,320);
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    private MenuItem mItemPreviewRGBA;
    private MenuItem mItemPreviewGray;
    private MenuItem mItemPreviewCanny;
    private MenuItem mItemPreviewFeatures;
    private Mat mRgba;
    private Mat mIntermediateMat;
    private Mat mIntermediateMat2;

    private Mat mGray;
    private int mViewMode;
    private static final int       VIEW_MODE_RGBA     = 0;
    private static final int       VIEW_MODE_GRAY     = 1;
    private static final int       VIEW_MODE_CANNY    = 2;
    private static final int VIEW_MODE_CONTOURS = 5;
    private Mat mCloseMorph;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.i(TAG, "onCreate");


        // Check/Request permission to use Camera
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) !=
                PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA},
                    REQUEST_CAMERA_PERMISSION);
        }

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.opencv_layout);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.openCvView);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        mItemPreviewRGBA = menu.add("Preview RGBA");
        mItemPreviewGray = menu.add("Preview GRAY");
        mItemPreviewCanny = menu.add("Canny");
        mItemPreviewFeatures = menu.add("Find features");
        return super.onCreateOptionsMenu(menu);
    }

    @Override
    protected void onPause() {
        super.onPause();
        Log.i(TAG, "onPause");
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.i(TAG, "onDestroy");
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    protected void onResume() {
        super.onResume();
        Log.i(TAG, "onResume");
        Log.i(TAG, "Trying to load OpenCV Asynchronously...");
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_6, this, mLoaderCallback);
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        Log.i(TAG, "CameraView Start");
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mIntermediateMat = new Mat(height, width, CvType.CV_8UC4);
        mIntermediateMat2 = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);

        mCloseMorph = Imgproc.getStructuringElement(Imgproc.CV_SHAPE_ELLIPSE, new Size(10, 10));
    }

    @Override
    public void onCameraViewStopped() {
        Log.i(TAG, "CameraView Stop");
        mRgba.release();
        mGray.release();
        mIntermediateMat.release();
        mIntermediateMat2.release();
        mCloseMorph.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        final int viewMode = mViewMode;
        switch (viewMode) {
            case VIEW_MODE_GRAY:
                // input frame has gray scale format
                Imgproc.cvtColor(inputFrame.gray(), mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
                break;
            case VIEW_MODE_RGBA:
                // input frame has RBGA format
                mRgba = inputFrame.rgba();
                break;
            case VIEW_MODE_CANNY:
                // input frame has gray scale format
                mRgba = inputFrame.rgba();
                Imgproc.Canny(inputFrame.gray(), mIntermediateMat, 80, 100);
                Imgproc.cvtColor(mIntermediateMat, mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
                break;
            case VIEW_MODE_CONTOURS:
                // Update rgba fresh
                mRgba = inputFrame.rgba();

                // blur grayscale, kernel size 3x3
                // TODO: 4/17/2016 Bilateral filter instead
//                Imgproc.resize(inputFrame.gray(), mSmall);
                Imgproc.blur(inputFrame.gray(), mGray, new Size(3,3));

                // Canny edges
                Imgproc.Canny(mGray, mIntermediateMat, 250, 256);

                // Closing Morphology operation
                Imgproc.morphologyEx(mIntermediateMat, mIntermediateMat2, Imgproc.MORPH_CLOSE, mCloseMorph);

                // Display closed grayscale image for now
//                Imgproc.cvtColor(mIntermediateMat2, mRgba, Imgproc.COLOR_GRAY2RGBA, 4);

                List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
                Mat hierarchy = new Mat();
                Imgproc.findContours(mIntermediateMat2, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

                // If contours were found (some edges exist) draw best contour onto image
                if (contours.size() > 0) {
                    double best_area = 0;
                    int best_idx = 0;

                    // Only need the largest area and index
                    int i = 0;
                    for (MatOfPoint contour : contours) {
                        double area = Imgproc.contourArea(contour);
                        if (area > best_area) {
                            best_area = area;
                            best_idx = i;
                        }
                        i++;
                    }

                    // Find minimum area rectangle enclosing largest contour
                    MatOfPoint best_contour = contours.get(best_idx);
                    RotatedRect best_rect = Imgproc.minAreaRect(new MatOfPoint2f(best_contour.toArray()));
                    Mat box = new Mat();
                    Imgproc.boxPoints(best_rect, box);

                    // Find bounding rectangle
                    Rect bbox = Imgproc.boundingRect(best_contour);

                    // Find and draw hough lines
                    Mat sub_edges = mIntermediateMat.submat(bbox);
                    Mat lines = new Mat();
                    Imgproc.HoughLinesP(sub_edges, lines, 1, Math.PI/180,
                            HOUGH_ACCUMULATOR_THRESHOLD, 20, 320);


//                    for (i = 0; i < lines.rows(); i++)
//                    {
//                        double[] line = lines.get(i,0);
//                        // Draw green lines
//                        Imgproc.line(mRgba,
//                                new Point(line[0]+bbox.x, line[1]+bbox.y),
//                                new Point(line[2]+bbox.x, line[3]+bbox.y),
//                                new Scalar(0,255,0),2
//                                );
//                    }

                    // get pruned list
                    ArrayList<double[]> new_list = new ArrayList<>();
                    ArrayList<Double> thetas = new ArrayList<Double>();
                    ArrayList<Double> rhos = new ArrayList<Double>();
                    pruneLines(lines, new_list, thetas, rhos, THETA_THRESHOLD, RHO_THRESHOLD);

                    for (i = 0; i < new_list.size(); i++)
                    {
                        double[] line = new_list.get(i);
                        // Draw green lines
                        Imgproc.line(mRgba,
                                new Point(line[0]+bbox.x, line[1]+bbox.y),
                                new Point(line[2]+bbox.x, line[3]+bbox.y),
                                new Scalar(0,255,0),2
                        );
                    }

                    // Draw contour onto rgba image
                    // Draw full contour in blue
                    Imgproc.drawContours(mRgba, contours, best_idx, new Scalar(0,0,255), 1);

                    // Draw bounding box in red
                    // Get Min Rect bounding box points
                    MatOfPoint box_points = new MatOfPoint(
                            new Point(box.get(0,0)[0], box.get(0,1)[0]),
                            new Point(box.get(1,0)[0], box.get(1,1)[0]),
                            new Point(box.get(2,0)[0], box.get(2,1)[0]),
                            new Point(box.get(3,0)[0], box.get(3,1)[0])
                    );

                    List<MatOfPoint> box_contours = new ArrayList<MatOfPoint>();
                    box_contours.add(box_points);

                    Imgproc.drawContours(mRgba, box_contours, 0, new Scalar(255,0,0), 2);

                    // Get BBox in green
                    // Get bounding box points
                    MatOfPoint bbox_points = new MatOfPoint(
                            bbox.tl(),
                            new Point(bbox.br().x, bbox.tl().y),
                            bbox.br(),
                            new Point(bbox.tl().x, bbox.br().y)
                    );
                    box_contours.add(bbox_points);
                    Imgproc.drawContours(mRgba, box_contours, 1, new Scalar(0,255,0), 2);
                }



                break;
        }

        return mRgba;
    }

    /* Prune lines that are too close in theta/rho to previous lines */
    private void pruneLines(Mat lines,
                            ArrayList<double[]> new_list,
                            ArrayList<Double> thetas,
                            ArrayList<Double> rhos,
                            double theta_threshold,
                            double rho_threshold) {

        for (int i=0; i < lines.rows(); i++)
        {
            double[] line = lines.get(i,0);
            double x1 = line[0];
            double y1 = line[1];
            double x2 = line[2];
            double y2 = line[3];

            double theta = Math.atan2(x2-x1, y2-y1);
            double rho = x1 * Math.cos(theta) + y1 * Math.sin(theta);

            // Check duplicate status
            boolean is_duplicate = false;
            for (double other_theta : thetas) {
                if (Math.abs(theta - other_theta) < theta_threshold) {
                    for (double other_rho : rhos) {
                        if (Math.abs(rho - other_rho) < rho_threshold) {
                            is_duplicate = true;
                            break;
                        }
                    }
                }
                if (is_duplicate)
                    break;
            }

            if (!is_duplicate) {
                thetas.add(theta);
                rhos.add(rho);
                new_list.add(line);
            }

        }

    }

    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);

        if (item == mItemPreviewRGBA) {
            mViewMode = VIEW_MODE_RGBA;
        } else if (item == mItemPreviewGray) {
            mViewMode = VIEW_MODE_GRAY;
        } else if (item == mItemPreviewCanny) {
            mViewMode = VIEW_MODE_CANNY;
        } else if (item == mItemPreviewFeatures) {
            mViewMode = VIEW_MODE_CONTOURS;
        }

        return true;
    }

//    public native void FindFeatures(long matAddrGr, long matAddrRgba);
}
