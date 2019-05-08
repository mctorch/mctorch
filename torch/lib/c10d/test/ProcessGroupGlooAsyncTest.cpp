#include <gloo/transport/tcp/device.h>

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAMultiStreamGuard.h>

#include <c10d/FileStore.hpp>
#include <c10d/ProcessGroupGloo.hpp>
#include <c10d/test/CUDATest.hpp>
#include <c10d/test/TestUtils.hpp>

using namespace c10d::test;

using at::cuda::CUDAStream;
using c10d::ProcessGroup;

template <typename T, typename... Args>
std::vector<T> initialize(const std::string& path, int N, Args&&... args) {
  std::vector<T> tests;
  for (auto i = 0; i < N; i++) {
    tests.push_back(std::move(T(path, std::forward<Args>(args)...)));
  }

  std::vector<std::thread> threads;
  for (auto i = 0; i < N; i++) {
    threads.push_back(std::thread([i, N, &tests] { tests[i].start(i, N); }));
  }

  for (auto& thread : threads) {
    thread.join();
  }

  return tests;
}

class AsyncTest {
 public:
  AsyncTest(const std::string& path) : path_(path) {}

  AsyncTest(AsyncTest&& other) {
    path_ = std::move(other.path_);
    pg_ = std::move(other.pg_);
  }

  ::c10d::ProcessGroupGloo& getProcessGroup() {
    return *pg_;
  }

  void start(int rank, int size) {
    auto store = std::make_shared<::c10d::FileStore>(path_, size);

    // Use tiny timeout to make this test run fast
    ::c10d::ProcessGroupGloo::Options options;
    options.timeout = std::chrono::milliseconds(50);
    ::gloo::transport::tcp::attr attr;
    options.devices.push_back(::gloo::transport::tcp::CreateDevice(attr));

    pg_ = std::unique_ptr<::c10d::ProcessGroupGloo>(
        new ::c10d::ProcessGroupGloo(store, rank, size, options));
  }

 protected:
  std::string path_;
  std::unique_ptr<::c10d::ProcessGroupGloo> pg_;
};

class AsyncInputIsOutputTest : public AsyncTest {
 public:
  AsyncInputIsOutputTest(const std::string& path, int numTensors)
      : AsyncTest(path),
        numTensors_(numTensors),
        numDevices_(cudaNumDevices()),
        state_(::at::globalContext().lazyInitCUDA()) {
    // Allocate inputs on available devices in a round robin fashion.
    inputs_.resize(numTensors_);
    for (auto i = 0; i < numTensors_; i++) {
      inputs_[i] = at::empty(
          {16, 16},
          at::device(
              {at::kCUDA, static_cast<c10::DeviceIndex>(i % numDevices_)}));
    }

    // Allocate a stream per device.
    //
    // The "current stream" is set globally per device in THC, so we
    // can't make two tensors on the same device use different streams
    // and pass this along to the collective (since it uses the THC
    // getters to retrieve the current stream).
    //
    at::cuda::OptionalCUDAGuard deviceGuard;
    streams_.reserve(numDevices_);
    for (auto i = 0; i < numDevices_; i++) {
      deviceGuard.set_index(i);
      streams_.push_back(at::cuda::getStreamFromPool());
    }
  }

  void wait(std::shared_ptr<ProcessGroup::Work>& work) {
    at::cuda::CUDAMultiStreamGuard guard(streams_);
    work->wait();
  }

  std::vector<at::Tensor> getTensors() {
    std::vector<at::Tensor> outputs(numTensors_);

    // For the duration of this function, make THC use our streams
    at::cuda::CUDAMultiStreamGuard guard(streams_);

    // Copy inputs to outputs
    for (auto i = 0; i < numTensors_; i++) {
      outputs[i] = inputs_[i].cpu();
    }

    return outputs;
  }

 protected:
  const int numTensors_;
  const int numDevices_;
  THCState* state_;
  std::vector<at::Tensor> inputs_;
  std::vector<CUDAStream> streams_;
};

class AsyncAllreduceTest : public AsyncInputIsOutputTest {
 public:
  AsyncAllreduceTest(const std::string& path, int numTensors)
      : AsyncInputIsOutputTest(path, numTensors) {}

  std::shared_ptr<c10d::ProcessGroup::Work> run() {
    // For the duration of this function, make THC use our streams
    at::cuda::CUDAMultiStreamGuard guard(streams_);

    // Launch sleep on every stream
    at::cuda::OptionalCUDAGuard deviceGuard;
    for (auto i = 0; i < numDevices_; i++) {
      deviceGuard.set_index(i);
      cudaSleep(streams_[i], 10 * 1000 * 1000);
    }

    // Launch value initialization for every tensor
    for (auto i = 0; i < numTensors_; i++) {
      deviceGuard.set_index(i % numDevices_);
      inputs_[i].fill_(pg_->getRank() * numTensors_ + i);
    }

    return pg_->allreduce(inputs_);
  }
};

class AsyncBroadcastTest : public AsyncInputIsOutputTest {
 public:
  AsyncBroadcastTest(const std::string& path, int numTensors)
      : AsyncInputIsOutputTest(path, numTensors) {}

  std::shared_ptr<c10d::ProcessGroup::Work> run(int rootRank, int rootTensor) {
    // For the duration of this function, make THC use our streams
    at::cuda::CUDAMultiStreamGuard guard(streams_);

    // Launch sleep on every stream
    at::cuda::OptionalCUDAGuard deviceGuard;
    for (auto i = 0; i < numDevices_; i++) {
      deviceGuard.set_index(i);
      cudaSleep(streams_[i], 10 * 1000 * 1000);
    }

    // Launch value initialization for every tensor
    for (auto i = 0; i < numTensors_; i++) {
      deviceGuard.set_index(i % numDevices_);
      inputs_[i].fill_(pg_->getRank() * numTensors_ + i);
    }

    ::c10d::BroadcastOptions options;
    options.rootRank = rootRank;
    options.rootTensor = rootTensor;
    return pg_->broadcast(inputs_, options);
  }
};

void runAsyncAllreduceTest(
    const std::string& path,
    size_t numProcesses,
    size_t numTensors) {
  auto tests = initialize<AsyncAllreduceTest>(path, numProcesses, numTensors);
  std::vector<std::shared_ptr<c10d::ProcessGroup::Work>> work(numProcesses);
  for (size_t i = 0; i < numProcesses; i++) {
    work[i] = tests[i].run();
  }

  // Wait for work to complete
  for (size_t i = 0; i < numProcesses; i++) {
    tests[i].wait(work[i]);
  }

  // Check results
  for (size_t i = 0; i < numProcesses; i++) {
    const auto size = numProcesses * numTensors;
    const auto expected = (size * (size - 1)) / 2;
    auto tensors = tests[i].getTensors();
    for (size_t j = 0; j < tensors.size(); j++) {
      auto& tensor = tensors[j];
      auto data = tensor.data<float>();
      for (auto k = 0; k < tensor.numel(); k++) {
        if (data[k] != expected) {
          throw std::runtime_error("BOOM!");
        }
      }
    }
  }
}

void runAsyncBroadcastTest(
    const std::string& path,
    size_t numProcesses,
    size_t numTensors) {
  auto tests = initialize<AsyncBroadcastTest>(path, numProcesses, numTensors);

  // Try every permutation of root rank and root tensor
  for (size_t rootRank = 0; rootRank < numProcesses; rootRank++) {
    for (size_t rootTensor = 0; rootTensor < numTensors; rootTensor++) {
      std::vector<std::shared_ptr<c10d::ProcessGroup::Work>> work(numProcesses);
      for (size_t i = 0; i < numProcesses; i++) {
        work[i] = tests[i].run(rootRank, rootTensor);
      }

      // Wait for work to complete
      for (size_t i = 0; i < numProcesses; i++) {
        tests[i].wait(work[i]);
      }

      // Check results
      const auto expected = (rootRank * numTensors + rootTensor);
      for (size_t i = 0; i < numProcesses; i++) {
        auto tensors = tests[i].getTensors();
        for (size_t j = 0; j < tensors.size(); j++) {
          auto& tensor = tensors[j];
          auto data = tensor.data<float>();
          for (auto k = 0; k < tensor.numel(); k++) {
            if (data[k] != expected) {
              throw std::runtime_error("BOOM!");
            }
          }
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  {
    TemporaryFile file;
    runAsyncAllreduceTest(file.path, 4, 2);
  }

  {
    TemporaryFile file;
    runAsyncBroadcastTest(file.path, 4, 1);
  }
  std::cout << "Test successful" << std::endl;
}
