#pragma once

#include <opencv2/core.hpp>

using Plane = cv::Vec4f;

struct CameraIntrinsics {
  float f;
  float cx;
  float cy;
};

cv::Mat3f normals_from_depth(const cv::Mat1f &depth, CameraIntrinsics intrinsics,
                             int window_size = 15, float max_rel_depth_diff = 0.1);

cv::Mat3b normals_to_rgb(const cv::Mat3f &normals);

cv::Mat1f get_surrounding_points(const cv::Mat1f &depth, int i, int j, CameraIntrinsics intrinsics,
                                 int window_size, float max_rel_depth_diff);

cv::Vec3f fit_plane(const cv::Mat &points);
