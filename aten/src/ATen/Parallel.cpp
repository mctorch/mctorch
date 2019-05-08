#include <ATen/Parallel.h>

#include <atomic>

#ifdef TH_BLAS_MKL
#include <mkl.h>
#endif

namespace at {

namespace {
// Number of threads set by the user
std::atomic<int> num_threads(-1);
}

void init_num_threads() {
  auto nthreads = num_threads.load();
  if (nthreads > 0) {
    set_num_threads(nthreads);
  } else {
#if defined(_OPENMP) && defined(TH_BLAS_MKL)
  // If we are using MKL an OpenMP make sure the number of threads match.
  // Otherwise, MKL and our OpenMP-enabled functions will keep changing the
  // size of the OpenMP thread pool, resulting in worse performance (and memory
  // leaks in GCC 5.4)
  omp_set_num_threads(mkl_get_max_threads());
#endif
  }
}

void set_num_threads(size_t nthreads) {
  if (nthreads == 0) {
    return;
  }
  num_threads.store(nthreads);
#ifdef _OPENMP
  omp_set_num_threads(nthreads);
#endif
#ifdef TH_BLAS_MKL
  mkl_set_num_threads(nthreads);

  // because PyTorch uses OpenMP outside of MKL invocations
  // as well, we want this flag to be false, so that
  // threads aren't destroyed and recreated across every
  // MKL / non-MKL boundary of OpenMP usage
  // See https://github.com/pytorch/pytorch/issues/13757
  mkl_set_dynamic(false);
#endif
}

// Explicitly calling omp_get_max_threads() as the size of the parallel
// region might be different in the new thread;
// Use init_num_threads() during thread initialization to ensure
// consistent size of parallel region in different threads
size_t get_num_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

PTThreadPool::PTThreadPool(
    std::size_t pool_size,
    int numa_node_id)
    : c10::ThreadPool(pool_size, numa_node_id) {}

void PTThreadPool::init_thread() {
  c10::setThreadName("PTThreadPool");
  at::init_num_threads();
}

namespace {

std::shared_ptr<TaskThreadPoolBase> createC10ThreadPool(
    int device_id,
    int pool_size,
    bool create_new) {
  static std::shared_ptr<TaskThreadPoolBase> pool =
      std::make_shared<PTThreadPool>(pool_size);
  // For now, the only accepted device id is 0
  // for the JIT inter-op pool (CPU),
  AT_ASSERT(device_id == 0);
  // we use the shared thread pool
  AT_ASSERT(!create_new);
  // and the size does not change
  AT_ASSERT(pool->size() == pool_size);
  return pool;
}

} // namespace

C10_REGISTER_CREATOR(ThreadPoolRegistry, C10, createC10ThreadPool);

} // namespace at
