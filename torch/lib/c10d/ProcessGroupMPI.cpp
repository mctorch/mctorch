#include <c10d/ProcessGroupMPI.hpp>

#include <map>

#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h> // Needed for CUDA-aware check
#endif

namespace c10d {

#define MPI_CHECK(cmd)                                                   \
  do {                                                                   \
    int mpiStatus = cmd;                                                 \
    if (mpiStatus != MPI_SUCCESS) {                                      \
      std::string err = "MPI error in: " + std::string(__FILE__) + ":" + \
          std::to_string(__LINE__) +                                     \
          ", with error code: " + std::to_string(mpiStatus);             \
      throw std::runtime_error(err);                                     \
    }                                                                    \
  } while (0)

namespace {

// Op mapping
std::map<ReduceOp, MPI_Op> mpiOp = {
    {ReduceOp::MIN, MPI_MIN},
    {ReduceOp::MAX, MPI_MAX},
    {ReduceOp::SUM, MPI_SUM},
    {ReduceOp::PRODUCT, MPI_PROD},
};
// Type mapping
std::map<at::ScalarType, MPI_Datatype> mpiDatatype = {
    {at::kByte, MPI_UNSIGNED_CHAR},
    {at::kChar, MPI_CHAR},
    {at::kDouble, MPI_DOUBLE},
    {at::kFloat, MPI_FLOAT},
    {at::kInt, MPI_INT},
    {at::kLong, MPI_LONG},
    {at::kShort, MPI_SHORT},
};

// Checking CUDA-aware MPI support, currently we only support CUDA aware
// MPI ops through Open MPI
bool cudaAwareMpiCheck() {
// Run time check
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  if (MPIX_Query_cuda_support() == 1) {
    return true;
  } else {
    return false;
  }
#else // !defined(MPIX_CUDA_AWARE_SUPPORT)
  return false;
#endif // MPIX_CUDA_AWARE_SUPPORT
}

// Checking the input tensor's validity
void checkSingleTensorHelper(const at::Tensor& tensor) {
  if (!tensor.is_contiguous()) {
    throw std::runtime_error("input tensor has to be contiguous");
  }
  if (tensor.is_sparse()) {
    throw std::runtime_error("input tensor has to be dense");
  }
  if (tensor.is_cuda() && !cudaAwareMpiCheck()) {
    throw std::runtime_error(
        "CUDA tensor detected and the MPI used doesn't "
        "have CUDA-aware MPI support");
  }
}

void checkSingleTensor(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() != 1) {
    throw std::runtime_error(
        "MPI process group does not support multi-GPU collectives");
  }
  checkSingleTensorHelper(tensors[0]);
}

void checkSameSizeAndType(
    const at::Tensor& tensor,
    const std::vector<at::Tensor>& tensors) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    if ((tensors[i].numel() != tensor.numel()) ||
        (tensors[i].type() != tensor.type())) {
      throw std::runtime_error("Tensors are not equal in size or data type");
    }
    checkSingleTensorHelper(tensors[i]);
  }
}

} // namespace

ProcessGroupMPI::AsyncWork::AsyncWork(at::Tensor tensor, MPI_Request request)
    : tensor_(std::move(tensor)), request_(request) {
  memset(&status_, 0, sizeof(status_));
}

ProcessGroupMPI::AsyncWork::~AsyncWork() {
  if (request_ != MPI_REQUEST_NULL) {
    std::cerr
        << "Attempted destruction of AsyncWork before work has completed, "
        << "terminating the program." << std::endl;
    std::terminate();
  }
}

bool ProcessGroupMPI::AsyncWork::isCompleted() {
  if (request_ == MPI_REQUEST_NULL) {
    return true;
  }

  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  int flag = 0;
  MPI_CHECK(MPI_Test(&request_, &flag, &status_));
  if (request_ != MPI_REQUEST_NULL) {
    return false;
  }

  // request_ == MPI_REQUEST_NULL; the work has completed
  // Populate exception if request was not successful
  if (status_.MPI_ERROR != MPI_SUCCESS) {
    populateException();
  }

  return true;
}

bool ProcessGroupMPI::AsyncWork::isSuccess() const {
  if (request_ != MPI_REQUEST_NULL) {
    throw std::runtime_error(
        "Invalid call to AsyncWork::isSuccess before work has completed");
  }

  return status_.MPI_ERROR == MPI_SUCCESS;
}

int ProcessGroupMPI::AsyncWork::sourceRank() const {
  return status_.MPI_SOURCE;
}

void ProcessGroupMPI::AsyncWork::wait() {
  if (request_ == MPI_REQUEST_NULL) {
    return;
  }

  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  MPI_CHECK(MPI_Wait(&request_, &status_));
  auto ok = (status_.MPI_ERROR == MPI_SUCCESS);
  if (!ok) {
    populateException();
    std::rethrow_exception(exception_);
  }
}

void ProcessGroupMPI::AsyncWork::populateException() {
  std::array<char, MPI_MAX_ERROR_STRING> buf;
  int len = buf.size();
  MPI_CHECK(MPI_Error_string(status_.MPI_ERROR, buf.data(), &len));
  exception_ =
      std::make_exception_ptr(std::runtime_error(std::string(buf.data(), len)));
}

// Static global states
int ProcessGroupMPI::mpiThreadSupport_ = 0;
std::mutex ProcessGroupMPI::pgGlobalMutex_;
// We only want to initialize once
std::once_flag ProcessGroupMPI::onceFlagInitMPI;

void ProcessGroupMPI::mpiExit() {
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  MPI_CHECK(MPI_Finalize());
}

void ProcessGroupMPI::initMPIOnce() {
  // Initialize MPI environment
  std::call_once(onceFlagInitMPI, []() {
    MPI_CHECK(MPI_Init_thread(
        nullptr, nullptr, MPI_THREAD_SERIALIZED, &mpiThreadSupport_));
    if (mpiThreadSupport_ < MPI_THREAD_SERIALIZED) {
      throw std::runtime_error(
          "Used MPI implementation doesn't have the "
          "minimum level of threading support: "
          "MPI_THREAD_SERIALIZED. This is required by "
          "c10d package");
    }
    if (std::atexit(ProcessGroupMPI::mpiExit)) {
      throw std::runtime_error("Fail to register the MPI exit handler");
    }
  });
}

std::shared_ptr<ProcessGroupMPI> ProcessGroupMPI::createProcessGroupMPI(
    std::vector<int> ranks) {
  // Once initialization
  initMPIOnce();

  MPI_Comm groupComm = MPI_COMM_WORLD;
  int rank = -1;
  int size = -1;

  {
    std::lock_guard<std::mutex> globalLock(pgGlobalMutex_);

    // If no ranks are specified, assume we're creating the root group
    if (!ranks.empty()) {
      MPI_Group worldGroup;
      MPI_Group ranksGroup;
      MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &worldGroup));
      MPI_CHECK(
          MPI_Group_incl(worldGroup, ranks.size(), ranks.data(), &ranksGroup));
      MPI_CHECK(MPI_Comm_create(MPI_COMM_WORLD, ranksGroup, &groupComm));
      MPI_CHECK(MPI_Group_free(&worldGroup));
      MPI_CHECK(MPI_Group_free(&ranksGroup));
    }

    // Fetch rank and world size for this group (MPI_COMM_WORLD or new)
    if (groupComm != MPI_COMM_NULL) {
      MPI_CHECK(MPI_Comm_rank(groupComm, &rank));
      MPI_CHECK(MPI_Comm_size(groupComm, &size));

      if (rank < 0 || size < 0) {
        throw std::runtime_error("Failed to get the world_size / rank");
      }
    }
  }

  // If this process is not part of the group, we don't construct a
  // process group instance. This is in line with the semantics of the
  // other process group types.
  if (groupComm == MPI_COMM_NULL) {
    return std::shared_ptr<ProcessGroupMPI>();
  }

  return std::make_shared<ProcessGroupMPI>(rank, size, groupComm);
}

ProcessGroupMPI::ProcessGroupMPI(int rank, int size, MPI_Comm pgComm)
    : ProcessGroup(rank, size), stop_(false), pgComm_(pgComm) {
  if (pgComm_ == MPI_COMM_NULL) {
    throw std::runtime_error("pgComm_ must not be MPI_COMM_NULL");
  }

  // Start the worker thread accepting MPI calls
  workerThread_ = std::thread(&ProcessGroupMPI::runLoop, this);
}

ProcessGroupMPI::~ProcessGroupMPI() {
  destroy();
}

void ProcessGroupMPI::destroy() {
  std::unique_lock<std::mutex> lock(pgMutex_);
  queueConsumeCV_.wait(lock, [&] { return queue_.empty(); });

  // Queue is empty, signal stop
  stop_ = true;

  // Release lock to allow threads to terminate
  lock.unlock();
  queueProduceCV_.notify_all();

  // Join the single worker thread
  workerThread_.join();
}

void ProcessGroupMPI::abort() {
  destroy();
  MPI_Abort(pgComm_, EXIT_FAILURE);
}

void ProcessGroupMPI::runLoop() {
  std::unique_lock<std::mutex> lock(pgMutex_);

  while (!stop_) {
    if (queue_.empty()) {
      queueProduceCV_.wait(lock);
      continue;
    }

    auto workTuple = std::move(queue_.front());

    queue_.pop_front();

    auto& workEntry = std::get<0>(workTuple);
    auto& work = std::get<1>(workTuple);

    lock.unlock();
    queueConsumeCV_.notify_one();

    try {
      workEntry->run(workEntry);
      work->finish();
    } catch (...) {
      work->finish(std::current_exception());
    }

    lock.lock();
  }
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::enqueue(
    std::unique_ptr<WorkEntry> entry) {
  auto work = std::make_shared<WorkMPI>();
  std::unique_lock<std::mutex> lock(pgMutex_);
  queue_.push_back(std::make_tuple(std::move(entry), work));
  lock.unlock();
  queueProduceCV_.notify_one();
  return work;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  checkSingleTensor(tensors);
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Bcast(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            opts.rootRank,
            pgComm_));
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(&tensors, nullptr, std::move(runFunc)));
  return enqueue(std::move(entry));
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  checkSingleTensor(tensors);

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Allreduce(
            MPI_IN_PLACE,
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            mpiOp.at(opts.reduceOp),
            pgComm_));
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(&tensors, nullptr, std::move(runFunc)));
  return enqueue(std::move(entry));
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  checkSingleTensor(tensors);

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        auto dataPtr = (entry->src)[0].data_ptr();
        void* sendbuf = (rank_ == opts.rootRank) ? MPI_IN_PLACE : dataPtr;
        void* recvbuf = (rank_ == opts.rootRank) ? dataPtr : nullptr;

        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Reduce(
            sendbuf,
            recvbuf,
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            mpiOp.at(opts.reduceOp),
            opts.rootRank,
            pgComm_));
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(&tensors, nullptr, std::move(runFunc)));
  return enqueue(std::move(entry));
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  checkSingleTensor(inputTensors);
  if (outputTensors.size() != 1) {
    throw std::runtime_error(
        "MPI process group only supports a single "
        "tensor op");
  }
  if (static_cast<size_t>(size_) != outputTensors[0].size()) {
    throw std::runtime_error(
        "All gather: number of output tensors should equal "
        "to the world size");
  }

  checkSameSizeAndType(inputTensors[0], outputTensors[0]);

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        std::vector<at::Tensor>& outputDataVec = entry->dst;
        auto flatOutputTensor = newLikeFlat(outputDataVec);

        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Allgather(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            flatOutputTensor.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            pgComm_));

        for (size_t i = 0; i < outputDataVec.size(); ++i) {
          outputDataVec[i].copy_(flatOutputTensor[i]);
        }
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(&inputTensors, &outputTensors[0], std::move(runFunc)));
  return enqueue(std::move(entry));
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  checkSingleTensor(inputTensors);

  if (rank_ != opts.rootRank) {
    if (outputTensors.size() > 0) {
      throw std::runtime_error(
          "Gather: number of output tensors should be 0 "
          "for non-root");
    }
  } else {
    if (outputTensors.size() != 1) {
      throw std::runtime_error("Gather: multi-GPU collective is not supported");
    }
    if (static_cast<size_t>(size_) != outputTensors[0].size()) {
      throw std::runtime_error(
          "Gather: number of output tensors should equal "
          "to the world size");
    }
    checkSameSizeAndType(inputTensors[0], outputTensors[0]);
  }

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        void* recvbuf = nullptr;
        at::Tensor flatOutputTensor;

        if (rank_ == opts.rootRank) {
          flatOutputTensor = newLikeFlat(entry->dst);
          recvbuf = flatOutputTensor.data_ptr();
        }

        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Gather(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            recvbuf,
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            opts.rootRank,
            pgComm_));

        if (rank_ == opts.rootRank) {
          std::vector<at::Tensor>& outputDataVec = entry->dst;
          // copy the flattened output tensors to the outputs
          for (size_t i = 0; i < outputDataVec.size(); ++i) {
            outputDataVec.at(i).copy_(flatOutputTensor[i]);
          }
        }
      };

  if (rank_ == opts.rootRank) {
    auto entry = std::unique_ptr<WorkEntry>(
        new WorkEntry(&inputTensors, &outputTensors[0], std::move(runFunc)));
    return enqueue(std::move(entry));
  } else {
    auto entry = std::unique_ptr<WorkEntry>(
        new WorkEntry(&inputTensors, nullptr, std::move(runFunc)));
    return enqueue(std::move(entry));
  }
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  checkSingleTensor(outputTensors);

  if (rank_ != opts.rootRank) {
    if (inputTensors.size() > 0) {
      throw std::runtime_error(
          "Scatter: number of input tensors should be 0 "
          "for non-root");
    }
  } else {
    if (inputTensors.size() != 1) {
      throw std::runtime_error(
          "Scatter: multi-GPU collective is not supported");
    }
    if (static_cast<size_t>(size_) != inputTensors[0].size()) {
      throw std::runtime_error(
          "Scatter: number of input tensors should equal "
          "to the world size");
    }
    checkSameSizeAndType(outputTensors[0], inputTensors[0]);
  }

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->dst)[0];
        void* sendbuf = nullptr;
        at::Tensor flatInputTensor;

        if (rank_ == opts.rootRank) {
          std::vector<at::Tensor>& inputDataVec = entry->src;
          flatInputTensor = newLikeFlat(inputDataVec);
          sendbuf = flatInputTensor.data_ptr();

          // copy the input tensors to the flatten large send buffer
          for (size_t i = 0; i < inputDataVec.size(); ++i) {
            flatInputTensor[i].copy_(inputDataVec.at(i));
          }
        }

        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Scatter(
            sendbuf,
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            opts.rootRank,
            pgComm_));
      };

  if (rank_ == opts.rootRank) {
    auto entry = std::unique_ptr<WorkEntry>(
        new WorkEntry(&inputTensors[0], &outputTensors, std::move(runFunc)));
    return enqueue(std::move(entry));
  } else {
    auto entry = std::unique_ptr<WorkEntry>(
        new WorkEntry(nullptr, &outputTensors, std::move(runFunc)));
    return enqueue(std::move(entry));
  }
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  throw std::runtime_error("ProcessGroupMPI does not support reduce_scatter");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  checkSingleTensor(tensors);

  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;

  {
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Isend(
        tensor.data_ptr(),
        tensor.numel(),
        mpiDatatype.at(tensor.scalar_type()),
        dstRank,
        tag,
        pgComm_,
        &request));
  }

  return std::make_shared<AsyncWork>(tensor, request);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  checkSingleTensor(tensors);

  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;

  {
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Irecv(
        tensor.data_ptr(),
        tensor.numel(),
        mpiDatatype.at(tensor.scalar_type()),
        srcRank,
        tag,
        pgComm_,
        &request));
  }

  return std::make_shared<AsyncWork>(tensor, request);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  checkSingleTensor(tensors);

  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;

  {
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Irecv(
        tensor.data_ptr(),
        tensor.numel(),
        mpiDatatype.at(tensor.scalar_type()),
        MPI_ANY_SOURCE,
        tag,
        pgComm_,
        &request));
  }

  return std::make_shared<AsyncWork>(tensor, request);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::barrier(
    const BarrierOptions& opts) {
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [this](std::unique_ptr<WorkEntry>& entry) {
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Barrier(pgComm_));
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(nullptr, nullptr, std::move(runFunc)));
  return enqueue(std::move(entry));
}

} // namespace c10d
