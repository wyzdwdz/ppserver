#include "predictor.h"

#include <chrono>

void PpPredictor::InitializeCore(const std::string& engine)
{
  pointpillar::lidar::VoxelizationParameter vp{};
  vp.min_range = nvtype::Float3(-69.12f, -39.68f, -2.0f);
  vp.max_range = nvtype::Float3(69.12f, 39.68f, 16.0f);
  vp.voxel_size = nvtype::Float3(0.16f, 0.16f, 18.0f);
  vp.grid_size = vp.compute_grid_size(vp.max_range, vp.min_range, vp.voxel_size);
  vp.max_voxels = 80000;
  vp.max_points_per_voxel = 32;
  vp.max_points = 300000;
  vp.num_feature = 4;

  pointpillar::lidar::PostProcessParameter pp{};
  pp.min_range = vp.min_range;
  pp.max_range = vp.max_range;
  pp.feature_size = nvtype::Int2(vp.grid_size.x / 2, vp.grid_size.y / 2);

  pointpillar::lidar::CoreParameter param{};
  param.voxelization = vp;
  param.lidar_model = engine;
  param.lidar_post = pp;
  core_ = pointpillar::lidar::create_core(param);

  if (core_ == nullptr)
  {
    throw std::runtime_error("Failed to initialize PointPillar predictor core");
  }
}

std::vector<float> PpPredictor::Predict(const float* lidar_points, int num_points)
{
  using std::chrono::high_resolution_clock;
  using std::chrono::duration;
  
  auto t1 = high_resolution_clock::now();
  auto bboxes = core_->forward(lidar_points, num_points, stream_);
  auto t2 = high_resolution_clock::now();

  duration<double, std::milli> time = t2 - t1;

  std::vector<float> out;
  out.reserve(bboxes.size() * 9);

  for (auto& bbox : bboxes)
  {
    out.emplace_back(bbox.x);
    out.emplace_back(bbox.y);
    out.emplace_back(bbox.z);
    out.emplace_back(bbox.w);
    out.emplace_back(bbox.h);
    out.emplace_back(bbox.l);
    out.emplace_back(bbox.rt);
    out.emplace_back(static_cast<float>(bbox.id));
    out.emplace_back(bbox.score);
  }

  logger_->info("Predict one frame, get {} boxes, runtime {} ms", bboxes.size(), time.count());

  return out;
}
