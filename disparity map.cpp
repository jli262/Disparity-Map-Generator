// https://sourishghosh.com/2016/dense-disparity-maps-orb-descriptors/
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <fstream>
#include <vector>

using namespace cv;
using namespace std;

Mat img_left, img_right, img_left_disp, img_right_disp;
Mat img_left_desc, img_right_desc;
vector<KeyPoint> kpl, kpr;

