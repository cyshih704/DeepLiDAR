#include <string>
#include <tuple>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <pybind11/pybind11.h>

#include "surface_normal.h"

namespace py = pybind11;

using CameraIntrinsicsTuple = std::tuple<double, double, double>;

void normals_from_depth_wrapper(const std::string &depth_in_path,
                                const std::string &normals_out_path,
                                CameraIntrinsicsTuple intrinsics_tuple, int window_size,
                                float max_rel_depth_diff) {
  CameraIntrinsics intrinsics{};
  intrinsics.f  = std::get<0>(intrinsics_tuple);
  intrinsics.cx = std::get<1>(intrinsics_tuple);
  intrinsics.cy = std::get<2>(intrinsics_tuple);
  cv::Mat depth = cv::imread(depth_in_path, cv::IMREAD_UNCHANGED);
  if (depth.size().area() == 0) {
    throw std::runtime_error("Empty image");
  }
  if (depth.channels() != 1) {
    throw std::runtime_error("Not a single-channel depth image. Image has " +
                             std::to_string(depth.channels()) + " channels.");
  }
  depth.convertTo(depth, CV_32F);
  cv::Mat3f normals     = normals_from_depth(depth, intrinsics, window_size, max_rel_depth_diff);
  cv::Mat3b normals_rgb = normals_to_rgb(normals);
  cvtColor(normals_rgb, normals_rgb, cv::COLOR_RGB2BGR);
  imwrite(normals_out_path, normals_rgb);
}

PYBIND11_MODULE(surface_normal, m) {
  m.doc() = "";
  m.def("normals_from_depth", &normals_from_depth_wrapper, py::arg("depth_in_path"),
        py::arg("normals_out_path"), py::arg("intrinsics"), py::arg("window_size") = 15,
        py::arg("max_rel_depth_diff") = 0.1);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}