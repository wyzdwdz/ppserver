#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include <CLI/App.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <spdlog/async.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "predictor.h"

using namespace boost::interprocess;
namespace fs = std::filesystem;

class SharedMemory {
    public:
        SharedMemory() {
            shm_obj_ = shared_memory_object(open_or_create, "pp_shm", read_write);
            offset_t shm_size = 0;
            if (shm_obj_.get_size(shm_size) && shm_size == 0) {
                shm_obj_.truncate(100 * 1024 * 1024);
            }
            region_ = mapped_region(shm_obj_, read_write);
            std::memset(region_.get_address(), 0, region_.get_size());
            state_ = static_cast<uint8_t*>(region_.get_address());
            num_ = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(region_.get_address()) + 1);
            data_ = reinterpret_cast<float*>(static_cast<uint8_t*>(region_.get_address()) + 5);
        }
        ~SharedMemory() {
            shared_memory_object::remove("pp_shm");
        }
        
        SharedMemory(const SharedMemory&) = delete;
        SharedMemory& operator=(const SharedMemory&) = delete;

        inline uint8_t GetState() const {
            return *state_;
        }

        inline void SetState(uint8_t state) {
            std::memcpy(state_, &state, sizeof(state));
        }

        inline uint32_t GetNumber() const {
            return *num_;
        }

        inline void SetNumber(uint32_t number) {
            std::memcpy(num_, &number, sizeof(number));
        };

        inline std::vector<float> GetData(size_t size) const {
            std::vector<float> data;
            data.resize(size);
            std::memcpy(data.data(), data_, sizeof(float) * size);
            return data;
        };

        inline void SetData(std::vector<float> data) {
            std::memcpy(data_, data.data(), sizeof(float) * data.size());
        };

    private:
        shared_memory_object shm_obj_;
        mapped_region region_;

        uint8_t* state_;
        uint32_t* num_;
        float* data_;
};

int main(int argc, char** argv)
{
  CLI::App app{"Pointpillar server"};

  std::string plan;
  auto po = app.add_option("-p,--plan", plan, "TensorRT plan file");
  po->required();

  CLI11_PARSE(app, argc, argv);

  spdlog::init_thread_pool(512, 1);
  auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

  auto main_logger =
    std::make_shared<spdlog::async_logger>("MAIN", stdout_sink, spdlog::thread_pool());
  auto pred_logger =
    std::make_shared<spdlog::async_logger>("PRED", stdout_sink, spdlog::thread_pool());

  SharedMemory shm;
  PpPredictor predictor(pred_logger, plan);

  while (true)
  {
    if (shm.GetState() == 0) {
      continue;
    }

    auto num = shm.GetNumber();
    auto data = shm.GetData(num * 4);

    auto blocks = predictor.Predict(data.data(), num);
    auto num_blocks = blocks.size() / 9;

    shm.SetNumber(num_blocks);
    shm.SetData(blocks);
    shm.SetState(0);
  }
}