#pragma once
#ifndef PREDICTOR_H
#define PREDICTOR_H

#include <memory>
#include <span>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <spdlog/async.h>
#include <spdlog/spdlog.h>

#include <common/check.hpp>
#include <pointpillar/pointpillar.hpp>

struct RenderItems
{
  std::vector<float> points;
  std::vector<pointpillar::lidar::BoundingBox> boxes;
};

class PpPredictor
{
public:
  PpPredictor(std::shared_ptr<spdlog::async_logger> logger, const std::string& engine) :
    logger_(logger)
  {
    InitializeCore(engine);
    cudaStreamCreate(&stream_);
    logger_->info("Initialize predictor success");
  }

  ~PpPredictor()
  {
    checkRuntime(cudaStreamDestroy(stream_));
  }

  PpPredictor() = delete;

  PpPredictor(const PpPredictor&) = delete;
  PpPredictor& operator=(const PpPredictor&) = delete;

  PpPredictor(PpPredictor&&) = delete;
  PpPredictor& operator=(PpPredictor&&) = delete;

  std::vector<float> Predict(const float* lidar_points, int num_points);

private:
  std::shared_ptr<spdlog::async_logger> logger_;

  std::shared_ptr<pointpillar::lidar::Core> core_;

  cudaStream_t stream_;

  void InitializeCore(const std::string& engine);
};

#endif
