#include <cmath>

#include <opencv2/core.hpp>

#include "surface_normal.h"

using namespace cv;
using std::abs;

// Returns a Nx3 matrix of 3D points surrounding the i,j pixel within the window_size window.
Mat1f get_surrounding_points(const Mat1f &depth, int i, int j, CameraIntrinsics intrinsics,
                             int window_size, float max_rel_depth_diff) {
  float f_inv        = 1.f / intrinsics.f;
  float cx           = intrinsics.cx;
  float cy           = intrinsics.cy;
  float center_depth = depth.at<float>(i, j);
  Mat1f points(window_size * window_size, 3);
  int count = 0;
  for (int idx = 0; idx < window_size; idx++) {
    for (int idy = 0; idy < window_size; idy++) {
      int row = i - window_size / 2 + idx;
      int col = j - window_size / 2 + idy;
      if (row >= depth.rows || col >= depth.cols) {
        continue;
      }
      float z = depth.at<float>(row, col);
      if (z == 0) {
        continue;
      }
      if (abs(z - center_depth) > max_rel_depth_diff * center_depth) {
        continue;
      }

      points.at<float>(count, 0) = (col - cx) * z * f_inv;
      points.at<float>(count, 1) = (row - cy) * z * f_inv;
      points.at<float>(count, 2) = z;
      count++;
    }
  }
  return points(Rect(0, 0, 3, count));
}

// Fits a plane to a set of 3D points.
// Returns only the plane normal for efficiency.
Vec3f fit_plane(const Mat &points) {
  constexpr int ncols = 3;
  Mat cov, centroid;
  calcCovarMatrix(points, cov, centroid, CV_COVAR_ROWS | CV_COVAR_NORMAL, CV_32F);
  SVD svd(cov, SVD::MODIFY_A);
  // Assign plane coefficients by the singular vector corresponding to the smallest
  // singular value.
  Vec3f normal = normalize(Vec3f(svd.vt.row(ncols - 1)));
  // Plane plane;
  // plane[ncols] = 0;
  // for (int c = 0; c < ncols; c++) {
  //   plane[c] = svd.vt.at<float>(ncols - 1, c);
  //   plane[ncols] += plane[c] * centroid.at<float>(0, c);
  // }
  return normal;
}

Mat3f normals_from_depth(const Mat1f &depth, CameraIntrinsics intrinsics, int window_size,
                         float max_rel_depth_diff) {
  Mat3f normals = Mat::zeros(depth.size(), CV_32FC3);
  for (int i = 0; i < depth.rows; i++) {
    for (int j = 0; j < depth.cols; j++) {
      if (depth.at<float>(i, j) == 0) {
        continue;
      }

      Mat1f points =
          get_surrounding_points(depth, i, j, intrinsics, window_size, max_rel_depth_diff);

      if (points.rows < 3) {
        continue;
      }

      Vec3f normal    = fit_plane(points);
      Vec3f direction = Vec3f(j - intrinsics.cx, i - intrinsics.cy, intrinsics.f);
      if (direction.dot(normal) < 0) {
        normal *= -1;
      }
      normals.at<Vec3f>(i, j) = normal;
    }
  }
  return normals;
}

constexpr uint8_t f2b(float x) { return static_cast<uint8_t>(127.5 * (1 - x)); }

Mat3b normals_to_rgb(const Mat3f &normals) {
  Mat3b res = Mat::zeros(normals.size(), CV_8UC3);
  for (int i = 0; i < normals.rows; i++) {
    for (int j = 0; j < normals.cols; j++) {
      Vec3f normal = normals.at<Vec3f>(i, j);
      if (normal[0] == 0 && normal[1] == 0 && normal[2] == 0)
        continue;
      res.at<Vec3b>(i, j)[0] = f2b(normal[0]);
      res.at<Vec3b>(i, j)[2] = f2b(normal[1]);
      res.at<Vec3b>(i, j)[1] = f2b(normal[2]);
    }
  }
  return res;
}
