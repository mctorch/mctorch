#pragma once

#include <atomic>
#include <memory>
#include <numeric>

#include <c10/core/Backend.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/TensorTypeId.h>
#include <c10/core/TensorTypeIdRegistration.h>
#include <c10/core/CopyBytes.h>

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/Flags.h>
#include <c10/util/Logging.h>
#include <c10/util/python_stub.h>

// A global boolean variable to control whether we free memory when a Tensor
// is shrinked to a smaller size. As a result, a Tensor is always going to
// keep the memory allocated for its maximum capacity reshaped to so far.
//
// This parameter is respected "upper-case" methods which call Resize()
// (e.g., CopyFrom, ResizeLike); it is NOT respected by Tensor::resize_
// or ShrinkTo, both of which guarantee to never to free memory.
C10_DECLARE_bool(caffe2_keep_on_shrink);

// Since we can have high variance in blob memory allocated across different
// inputs in the same run, we will shrink the blob only if the memory gain
// is larger than this flag in bytes.  This only applies to functions which
// respect caffe2_keep_on_shrink.
C10_DECLARE_int64(caffe2_max_keep_on_shrink_memory);


namespace at {
class Tensor;
}

namespace c10 {
class Scalar;
struct Storage;

/**
 * A utility function to convert vector<int> to vector<int64_t>.
 */
inline std::vector<int64_t> ToVectorint64_t(ArrayRef<int> src) {
  return std::vector<int64_t>(src.begin(), src.end());
}

/**
 * Return product of all dimensions starting from k
 */
inline int64_t size_from_dim_(int k, IntArrayRef dims) {
  int64_t r = 1;
  for (size_t i = k; i < dims.size(); ++i) {
    r *= dims[i];
  }
  return r;
}

// Product of all dims up to k (not including dims[k])
inline int64_t size_to_dim_(int k, IntArrayRef dims) {
  AT_ASSERT((unsigned)k <= dims.size());
  int64_t r = 1;
  for (int i = 0; i < k; ++i) {
    r *= dims[i];
  }
  return r;
}

// Product of all dims between k and l (not including dims[k] and dims[l])
inline int64_t size_between_dim_(int k, int l, IntArrayRef dims) {
  AT_ASSERT((unsigned)l < dims.size());
  int64_t r = 1;
  if (k < l) {
    for (int i = k + 1; i < l; ++i) {
      r *= dims[i];
    }
  } else {
    for (int i = l + 1; i < k; ++i) {
      r *= dims[i];
    }
  }
  return r;
}

// Wrap around axis_index if it is negative, s.t., -1 is the last dim
inline int canonical_axis_index_(int axis_index, int ndims) {
  AT_ASSERT(axis_index >= -ndims);
  AT_ASSERT(axis_index < ndims);
  if (axis_index < 0) {
    return axis_index + ndims;
  }
  return axis_index;
}

using PlacementDtor = void (*)(void*, size_t);

/*
 * A Context that will call extra placement deleter during
 * deconstruction.
 *
 * Accept a already constructed DataPtr and store it as member
 * during destruction, we'll call extra deleter on the underlying
 * data pointer before the DataPtr is destructed.
 * `data_ptr_` owns the memory.
 */
struct C10_API PlacementDeleteContext {
  DataPtr data_ptr_;
  PlacementDtor placement_dtor_;
  size_t size_;
  PlacementDeleteContext(
      DataPtr&& data_ptr,
      PlacementDtor placement_dtor,
      size_t size)
      : data_ptr_(std::move(data_ptr)),
        placement_dtor_(placement_dtor),
        size_(size) {}
  static DataPtr makeDataPtr(
      DataPtr&& data_ptr,
      PlacementDtor placement_dtor,
      size_t size,
      Device device);
  ~PlacementDeleteContext() {
    placement_dtor_(data_ptr_.get(), size_);
    // original memory will be freed when data_ptr_ is destructed
  }
};

struct TensorImpl;

struct C10_API AutogradMetaInterface {
  virtual void set_requires_grad(bool requires_grad, at::TensorImpl* self_impl) = 0;
  virtual bool requires_grad() const = 0;
  virtual at::Tensor& grad() = 0;
  virtual const at::Tensor& grad() const = 0;
  virtual ~AutogradMetaInterface();
};

// NOTE [ Version Counter Sharing ]
//
// Every Tensor has a version counter. Version counters are incremented whenever the
// data or size of a tensor changes through in-place Variable operations. Version
// counters are used to detect modifications to saved variables which would result in
// incorrect gradient calculations. Version counters may be shared between Variables:
//
// 1. A view shares the version counter of the base Variable,
// 2. `x.detach()` shares the version counter of `x`,
// 3. Unpacked saved variables share the version counter of the source.
//
// Version counters are not shared in these scenarios:
//
// 1. When we replace a `Variable`'s underlying `Tensor` by calling `set_data(...)`,
// 2. `x.data` does not share the version counter of `x`. (See discussion at
// https://github.com/pytorch/pytorch/issues/5396)
//
// Question: Why do we put the version counter in TensorImpl instead of AutogradMeta?
//
// Answer: After the Variable/Tensor merge, a tensor will not have AutogradMeta when
// its `requires_grad_` is false, but when we use this tensor in the forward pass of
// a function that requires saving this tensor for backward, we need to keep track of
// this tensor's version to make sure it's always valid in the autograd graph.
//
// To achieve this goal, we put the version counter in TensorImpl instead of AutogradMeta,
// and have it always be available. This allows us to have the optimization of not
// carrying AutogradMeta when a tensor doesn't require gradient.
//
// A hypothetical alternative way to achieve this goal is to initialize AutogradMeta and
// create the version counter for the non-requires-grad tensor only when it's saved for
// backward. However, since saving a tensor for backward happens in the forward pass, and
// our invariant is that forward pass needs to be thread-safe, lazy-initializing AutogradMeta
// when saving a tensor can introduce race conditions when we are running the forward
// pass in multi-thread scenarios, thus making the forward pass not thread-safe anymore,
// which breaks the invariant.
struct C10_API VariableVersion {
 public:
  // NOTE: As of C++11 and 14, default-constructing a std::atomic variable
  // leaves it in a persistently undefined state. See
  // https://cplusplus.github.io/LWG/issue2334.
  VariableVersion(uint32_t version = 0)
      : version_block_(std::make_shared<std::atomic<uint32_t>>(version)) {}

  void bump() noexcept {
    version_block_->fetch_add(1);
  }

  uint32_t current_version() const noexcept {
    return version_block_->load();
  }

 private:
  std::shared_ptr<std::atomic<uint32_t>> version_block_;
};

/**
 * The low-level representation of a tensor, which contains a pointer
 * to a storage (which contains the actual data) and metadata (e.g., sizes and
 * strides) describing this particular view of the data as a tensor.
 *
 * Some basic characteristics about our in-memory representation of
 * tensors:
 *
 *  - It contains a pointer to a storage struct (Storage/StorageImpl)
 *    which contains the pointer to the actual data and records the
 *    data type and device of the view.  This allows multiple tensors
 *    to alias the same underlying data, which allows to efficiently
 *    implement differing *views* on a tensor.
 *
 *  - The tensor struct itself records view-specific metadata about
 *    the tensor, e.g., sizes, strides and offset into storage.
 *    Each view of a storage can have a different size or offset.
 *
 *  - This class is intrusively refcounted.  It is refcounted so that
 *    we can support prompt deallocation of large tensors; it is
 *    intrusively refcounted so that we can still perform reference
 *    counted operations on raw pointers, which is often more convenient
 *    when passing tensors across language boundaries.
 *
 *  - For backwards-compatibility reasons, a tensor may be in an
 *    uninitialized state.  A tensor may be uninitialized in the following
 *    two ways:
 *
 *      - A tensor may be DTYPE UNINITIALIZED.  A tensor of this
 *        form has an uninitialized dtype.  This situation most
 *        frequently arises when a user writes Tensor x(CPU).  The dtype and
 *        is subsequently initialized when mutable_data<T>() is
 *        invoked for the first time.
 *
 *      - A tensor may be STORAGE UNINITIALIZED.  A tensor of this form
 *        has non-zero size, but has a storage with a null data pointer.
 *        This situation most frequently arises when a user calls
 *        Resize() or FreeMemory().  This is because Caffe2 historically
 *        does lazy allocation: allocation of data doesn't occur until
 *        mutable_data<T>() is invoked.  A tensor with zero size is
 *        always storage initialized, because no allocation is necessary
 *        in this case.
 *
 *    All combinations of these two uninitialized states are possible.
 *    Consider the following transcript in idiomatic Caffe2 API:
 *
 *      Tensor x(CPU); // x is storage-initialized, dtype-UNINITIALIZED
 *      x.Resize(4); // x is storage-UNINITIALIZED, dtype-UNINITIALIZED
 *      x.mutable_data<float>(); // x is storage-initialized, dtype-initialized
 *      x.FreeMemory(); // x is storage-UNINITIALIZED, dtype-initialized.
 *
 *    All other fields on tensor are always initialized.  In particular,
 *    size is always valid. (Historically, a tensor declared as Tensor x(CPU)
 *    also had uninitialized size, encoded as numel == -1, but we have now
 *    decided to default to zero size, resulting in numel == 0).
 *
 *    Uninitialized storages MUST be uniquely owned, to keep our model
 *    simple.  Thus, we will reject operations which could cause an
 *    uninitialized storage to become shared (or a shared storage to
 *    become uninitialized, e.g., from FreeMemory).
 *
 *    In practice, tensors which are storage-UNINITIALIZED and
 *    dtype-UNINITIALIZED are *extremely* ephemeral: essentially,
 *    after you do a Resize(), you basically always call mutable_data()
 *    immediately afterwards.  Most functions are not designed to
 *    work if given a storage-UNINITIALIZED, dtype-UNINITIALIZED tensor.
 *
 *    We intend to eliminate all uninitialized states, so that every
 *    tensor is fully initialized in all fields.  Please do not write new code
 *    that depends on these uninitialized states.
 */
struct C10_API TensorImpl : public c10::intrusive_ptr_target {
  TensorImpl() = delete;

  /**
   * Construct a 1-dim 0-size tensor backed by the given storage.
   */
  TensorImpl(Storage&& storage, TensorTypeId type_id);

  /**
   * Construct a 1-dim 0 size tensor that doesn't have a storage.
   */
  TensorImpl(TensorTypeId type_id, const caffe2::TypeMeta& data_type, c10::optional<c10::Device> device_opt);

 private:
  // This constructor is private, because the data_type is redundant with
  // storage.  Still, we pass it in separately because it's easier to write
  // the initializer list if we're not worried about storage being moved out
  // from under us.
  TensorImpl(Storage&& storage, TensorTypeId type_id, const caffe2::TypeMeta& data_type, c10::optional<c10::Device>);

 public:
  TensorImpl(const TensorImpl&) = delete;
  TensorImpl& operator=(const TensorImpl&) = delete;
  TensorImpl(TensorImpl&&) = default;
  TensorImpl& operator=(TensorImpl&&) = default;

  /**
   * Release (decref) storage, and any other external allocations.  This
   * override is for `intrusive_ptr_target` and is used to implement weak
   * tensors.
   */
  virtual void release_resources() override;

  // TODO: Ideally, type_id() would be the *only* key we need to consult
  // to do a dispatch, instead of having to grovel through three different
  // variables.  Here's what's standing in the way:
  //
  //  - To eliminate ScalarType, we have to allocate a TensorTypeId for
  //    each ScalarType+Backend combination, and then set it appropriately
  //    when we initially allocate a TensorImpl.
  //
  //  - To eliminate is_variable, we have to allocate two classes of
  //    TensorTypeId: ones that are variables, and ones that are not.
  //    We may not want to eliminate this in the short term, because
  //    hard-coding variable status into type_id() makes it more difficult
  //    to do the "thread-local no_grad" trick (where we process Variables
  //    "as if" they were non-Variables by setting a thread local variable.)
  //
  // TODO: type() is a very attractive name for a method, but we don't
  // actually want people to use it.  Rename this to something else.

  /**
   * Return the TensorTypeId corresponding to this Tensor.  In the future,
   * this will be the sole piece of information required to dispatch
   * to an operator; however, at the moment, it is not used for
   * dispatch.
   *
   * type_id() and type() are NOT in one-to-one correspondence; we only
   * have a single type_id() for CPU tensors, but many Types (CPUFloatTensor,
   * CPUDoubleTensor...)
   */
  TensorTypeId type_id() const { return type_id_; }

  /**
   * Return a reference to the sizes of this tensor.  This reference remains
   * valid as long as the tensor is live and not resized.
   */
  virtual IntArrayRef sizes() const;

  /**
   * Return a reference to the strides of this tensor.  This reference remains
   * valid as long as the tensor is live and not restrided.
   */
  virtual IntArrayRef strides() const;

  /**
   * Return the number of dimensions of this tensor.  Note that 0-dimension
   * represents a Tensor that is a Scalar, e.g., one that has a single element.
   */
  virtual int64_t dim() const;

  /**
   * True if this tensor has storage. See storage() for details.
   */
  virtual bool has_storage() const;

  /**
   * Return the underlying storage of a Tensor.  Multiple tensors may share
   * a single storage.  A Storage is an impoverished, Tensor-like class
   * which supports far less operations than Tensor.
   *
   * Avoid using this method if possible; try to use only Tensor APIs to perform
   * operations.
   */
  virtual const Storage& storage() const;

  /**
   * The number of elements in a tensor.
   *
   * WARNING: Previously, if you were using the Caffe2 API, you could
   * test numel() == -1 to see if a tensor was uninitialized.  This
   * is no longer true; numel always accurately reports the product
   * of sizes of a tensor.
   */
  virtual int64_t numel() const {
#ifdef DEBUG
    AT_ASSERT(compute_numel() == numel_);
#endif
    return numel_;
  }

  /**
   * Whether or not a tensor is laid out in contiguous memory.
   *
   * Tensors with non-trivial strides are not contiguous.  See
   * compute_contiguous() for the exact definition of whether or not
   * a tensor is contiguous or not.
   */
  virtual bool is_contiguous() const {
#ifdef DEBUG
    AT_ASSERT(compute_contiguous() == is_contiguous_);
#endif
    return is_contiguous_;
  }

  bool is_sparse() const {
    // NB: This method is not virtual and avoid dispatches for performance reasons.
    auto tid = type_id();
    // NB: At the moment, variables have the same TensorTypeId as their
    // corresponding tensor, but if this ever changes, we need to modify this.
    return tid == SparseCPUTensorId() || tid == SparseCUDATensorId() || tid == SparseHIPTensorId();
  }

  bool is_quantized() const {
    // NB: This method is not virtual and avoid dispatches for performance reasons.
    auto tid = type_id();
    // NB: At the moment, variables have the same TensorTypeId as their
    // corresponding tensor, but if this ever changes, we need to modify this.
    return tid == QuantizedCPUTensorId();
  }

  bool is_cuda() const {
    // NB: This method is not virtual and avoid dispatches for performance reasons.
    auto tid = type_id();
    // NB: At the moment, variables have the same TensorTypeId as their
    // corresponding tensor, but if this ever changes, we need to modify this.
    return tid == CUDATensorId() || tid == SparseCUDATensorId();
  }

  bool is_hip() const {
    // NB: This method is not virtual and avoid dispatches for performance reasons.
    auto tid = type_id();
    // NB: At the moment, variables have the same TensorTypeId as their
    // corresponding tensor, but if this ever changes, we need to modify this.
    return tid == HIPTensorId() || tid == SparseHIPTensorId();
  }

  bool is_mkldnn() const {
    return type_id() == MkldnnCPUTensorId();
  }

  int64_t get_device() const {
    if (device_opt_.has_value()) {
      // See NOTE [c10::optional operator usage in CUDA]
      return (*device_opt_).index();
    }

    AT_ERROR(
        "tensor with backend ", toString(tensorTypeIdToBackend(type_id())),
        " does not have a device");
  }

  Device device() const {
    if (device_opt_.has_value()) {
      // See NOTE [c10::optional operator usage in CUDA]
      return *device_opt_;
    }

    AT_ERROR(
        "tensor with backend ", toString(tensorTypeIdToBackend(type_id())),
        " does not have a device");
  }

  Layout layout() const {
    // NB: This method is not virtual and avoid dispatches for perf.
    if (is_sparse()) {
      return kSparse;
    } else if (is_mkldnn()) {
      return kMkldnn;
    } else {
      return kStrided;
    }
  }

  /**
   * If `condition_when_zero_dim` is true, and the tensor is a 1-dim, 1-size
   * tensor, reshape the tensor into a 0-dim tensor (scalar).
   *
   * This helper function is called from generated wrapper code, to help
   * "fix up" tensors that legacy code didn't generate in the correct shape.
   * For example, suppose that we have a legacy function 'add' which produces
   * a tensor which is the same shape as its inputs; however, if the inputs
   * were zero-dimensional, it produced a 1-dim 1-size tensor (don't ask).
   * result->maybe_zero_dim(lhs->dim() == 0 && rhs->dim() == 0) will be called,
   * correctly resetting the dimension to 0 when when the inputs had 0-dim.
   *
   * As we teach more and more of TH to handle 0-dim correctly, this function
   * will become less necessary.  At the moment, it is often called from functions
   * that correctly handle the 0-dim case, and is just dead code in this case.
   * In the glorious future, this function will be eliminated entirely.
   */
  virtual TensorImpl* maybe_zero_dim(bool condition_when_zero_dim);

  /**
   * True if a tensor was auto-wrapped from a C++ or Python number.
   * For example, when you write 't + 2', 2 is auto-wrapped into a Tensor
   * with `is_wrapped_number_` set to true.
   *
   * Wrapped numbers do not participate in the result type computation for
   * mixed-type operations if there are any Tensors that are not wrapped
   * numbers.  This is useful, because we want 't + 2' to work with
   * any type of tensor, not just LongTensor (which is what integers
   * in Python represent).
   *
   * Otherwise, they behave like their non-wrapped equivalents.
   * See [Result type computation] in TensorIterator.h.
   *
   * Why did we opt for wrapped numbers, as opposed to just having
   * an extra function add(Tensor, Scalar)?  This helps greatly reduce
   * the amount of code we have to write for add, when actually
   * a Tensor-Scalar addition is really just a Tensor-Tensor
   * addition when the RHS is 0-dim (except for promotion behavior.)
   *
   * WARNING: It is NOT valid to call this method on a Variable.
   * See Note [We regret making Variable hold a Tensor]
   */
  bool is_wrapped_number() const {
    AT_ASSERT(!is_variable());  // TODO: remove this when Variable and Tensor are merged
    return is_wrapped_number_;
  }

  /**
   * Set whether or not a tensor was auto-wrapped from a C++ or Python
   * number.  You probably don't want to call this, unless you are
   * writing binding code.
   *
   * WARNING: It is NOT valid to call this method on a Variable.
   * See Note [We regret making Variable hold a Tensor]
   */
  void set_wrapped_number(bool value) {
    AT_ASSERT(!is_variable());  // TODO: remove this when Variable and Tensor are merged
    AT_ASSERT(dim() == 0);
    is_wrapped_number_ = value;
  }

  // ~~~~~ Autograd API ~~~~~
  // Some methods below are defined in TensorImpl.cpp because Tensor is an
  // incomplete type.
  //
  // Note [Tensor versus Variable in C++]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Autograd methods are only valid for the Variable::Impl subclass
  // of Tensor.  This is due to some questionable life choices, where
  // a Variable has a Tensor (so they are not the same thing), but
  // a Variable is a Tensor (they are subclassed, so that you can write
  // code on Tensor that works both with Variables and Tensors.  Poor
  // man's polymorphism).  Variable does NOT satisfy the Liskov Substitution
  // Principle for Tensor; generally you want to work with all Variables,
  // or all Tensors, but not a mix of both.  We intend to fix this in
  // the future.
  //
  // Note [We regret making Variable hold a Tensor]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Tensor has a bunch of fields in it.  Are those fields always valid?
  // Not necessarily: the Variable::Impl subclass of a tensor doesn't use these
  // fields; instead, it *forwards* them to a contained, inner tensor
  // (the 'data' tensor).  It doesn't even bother keeping the fields on the
  // outer tensor up-to-date, because an end user could grab the inner
  // tensor and directly, e.g., resize it (making any outer fields we track
  // stale).
  //
  // As you might imagine, this is a TERRIBLE state of affairs to be in.
  // It makes implementing everything on TensorImpl complicated: if
  // you directly access a field on TensorImpl, you must *virtualize*
  // the function, if you want it to work correctly when called from
  // Variable (because we need to override the method to avoid looking
  // in our fields, and look in the data tensor's fields.)  Anything that
  // isn't virtualized, won't work if called on a variable.
  //
  // The way to fix this is to make Variable::Impl stop holding a tensor;
  // instead, it should just *be* a tensor.

  /**
   * Set whether or not a tensor requires gradient.
   *
   * It is only valid to call this method on a Variable.
   * See Note [Tensor versus Variable in C++].
   */
  void set_requires_grad(bool requires_grad) {
    if (autograd_meta()) {
      autograd_meta()->set_requires_grad(requires_grad, this);
    } else {
      AT_ERROR("set_requires_grad is not implemented for Tensor");
    }
  }

  /**
   * True if a tensor requires gradient.  Tensors which require gradient
   * have history tracked for any operations performed on them, so that
   * we can automatically differentiate back to them.  A tensor that
   * requires gradient and has no history is a "leaf" tensor, which we
   * accumulate gradients into.
   *
   * It is only valid to call this method on a Variable.
   * See Note [Tensor versus Variable in C++].
   */
  bool requires_grad() const {
    if (autograd_meta()) {
      return autograd_meta()->requires_grad();
    } else {
      AT_ERROR("requires_grad is not implemented for Tensor");
    }
  }

  /**
   * Return a mutable reference to the gradient.  This is conventionally
   * used as `t.grad() = x` to set a gradient to a completely new tensor.
   *
   * It is only valid to call this method on a Variable.
   * See Note [Tensor versus Variable in C++].
   */
  at::Tensor& grad();

  /**
   * Return the accumulated gradient of a tensor.  This gradient is written
   * into when performing backwards, when this tensor is a leaf tensor.
   *
   * It is only valid to call this method on a Variable.
   * See Note [Tensor versus Variable in C++].
   */
  const at::Tensor& grad() const;

  /**
   * Return a typed data pointer to the actual data which this tensor refers to.
   * This checks that the requested type (from the template parameter) matches
   * the internal type of the tensor.
   *
   * It is invalid to call data() on a dtype-uninitialized tensor, even if
   * the size is 0.
   *
   * WARNING: If a tensor is not contiguous, you MUST use strides when
   * performing index calculations to determine the location of elements in
   * the tensor.  We recommend using 'TensorAccessor' to handle this computation
   * for you; this class is available from 'Tensor'.
   *
   * WARNING: It is NOT valid to call this method on a Variable.
   * See Note [We regret making Variable hold a Tensor]
   */
  template <typename T>
  inline T * data() const {
    AT_ASSERT(!is_variable());  // TODO: remove this when Variable and Tensor are merged
    AT_CHECK(has_storage(),
        "Cannot access data pointer of Tensor that doesn't have storage");
    AT_ASSERTM(
        storage_initialized(),
        "The tensor has a non-zero number of elements, but its data is not allocated yet. "
        "Caffe2 uses a lazy allocation, so you will need to call "
        "mutable_data() or raw_mutable_data() to actually allocate memory.");
    AT_ASSERTM(
        storage_.IsType<T>(),
        "Tensor type mismatch, caller expects elements to be ",
        caffe2::TypeMeta::TypeName<T>(),
        ", while tensor contains ",
        data_type_.name(),
        ". ");
    // We managed the type check ourselves
    return storage_.unsafe_data<T>() + storage_offset_;
  }

  /**
   * Return a void* data pointer to the actual data which this tensor refers to.
   *
   * It is invalid to call data() on a dtype-uninitialized tensor, even if the
   * size is 0.
   *
   * WARNING: The data pointed to by this tensor may not contiguous; do NOT
   * assume that itemsize() * numel() is sufficient to compute the bytes that
   * can be validly read from this tensor.
   *
   * WARNING: It is NOT valid to call this method on a Variable.
   * See Note [We regret making Variable hold a Tensor]
   */
  inline void* data() const {
    AT_ASSERT(!is_variable());  // TODO: remove this when Variable and Tensor are merged
    AT_CHECK(has_storage(),
        "Cannot access data pointer of Tensor that doesn't have storage");
    AT_ASSERT(dtype_initialized());
    return static_cast<void*>(
        static_cast<char*>(storage_.data()) +
        data_type_.itemsize() * storage_offset_);
  }

  /**
   * This is just like data(), except it works with Variables.
   * This function will go away once Variable and Tensor are merged.
   * See Note [We regret making Variable hold a Tensor]
   */
  virtual void* slow_data() const {
    return data();
  }

  /**
   * Like data<T>(), but performs no checks.  You are responsible for ensuring
   * that all invariants required by data() are upheld here.
   *
   * WARNING: It is NOT valid to call this method on a Variable.
   * See Note [We regret making Variable hold a Tensor]
   */
  template <typename T>
  inline T * unsafe_data() const {
    return storage_.unsafe_data<T>() + storage_offset_;
  }

  /**
   * Returns the TypeMeta of a tensor, which describes what data type
   * it is (e.g., int, float, ...)
   */
  const caffe2::TypeMeta& dtype() const {
    return data_type_;
  }

  /**
   * Return the size of a single element of this tensor in bytes.
   */
  size_t itemsize() const {
    AT_ASSERT(dtype_initialized());
    return data_type_.itemsize();
  }

  /**
   * Return the offset in number of elements into the storage that this
   * tensor points to.  Most tensors have storage_offset() == 0, but,
   * for example, an index into a tensor will have a non-zero storage_offset().
   *
   * WARNING: This is NOT computed in bytes.
   *
   * XXX: The only thing stopping this function from being virtual is Variable.
   */
  virtual int64_t storage_offset() const {
    return storage_offset_;
  }

  /**
   * True if a tensor has no elements (e.g., numel() == 0).
   */
  inline bool is_empty() const {
    return numel() == 0;
  }

  /**
   * Change the dimensionality of a tensor.  This is truly a resize:
   * old sizes, if they are still valid, are preserved (this invariant
   * is utilized by some call-sites, e.g., the implementation of squeeze, which
   * mostly wants the sizes to stay the same).  New dimensions are given zero
   * size and zero stride; this is probably not what you want--you should
   * set_size/set_stride afterwards.
   *
   * TODO: This should be jettisoned in favor of `set_sizes_and_strides`,
   * which is harder to misuse.
   */
  virtual void resize_dim(int64_t ndim) {
    AT_CHECK(allow_tensor_metadata_change(), "resize_dim is not allowed on Tensor created from .data or .detach()");
    sizes_.resize(ndim, 0);
    strides_.resize(ndim, 0);
    refresh_numel();
    refresh_contiguous();
  }

  /**
   * Change the size at some dimension.  This DOES NOT update strides;
   * thus, most changes to size will not preserve contiguity.  You probably
   * also want to call set_stride() when you call this.
   *
   * TODO: This should be jettisoned in favor of `set_sizes_and_strides`,
   * which is harder to misuse.
   */
  virtual void set_size(int64_t dim, int64_t new_size) {
    AT_CHECK(allow_tensor_metadata_change(), "set_size is not allowed on Tensor created from .data or .detach()");
    sizes_.at(dim) = new_size;
    refresh_numel();
    refresh_contiguous();
  }

  /**
   * Change the stride at some dimension.
   *
   * TODO: This should be jettisoned in favor of `set_sizes_and_strides`,
   * which is harder to misuse.
   */
  virtual void set_stride(int64_t dim, int64_t new_stride) {
    AT_CHECK(allow_tensor_metadata_change(), "set_stride is not allowed on Tensor created from .data or .detach()");
    strides_[dim] = new_stride;
    refresh_numel();
    refresh_contiguous();
  }

  /**
   * Set the offset into the storage of this tensor.
   *
   * WARNING: This does NOT check if the tensor is in bounds for the new
   * location at the storage; the caller is responsible for checking this
   * (and resizing if necessary.)
   */
  virtual void set_storage_offset(int64_t storage_offset) {
    AT_CHECK(allow_tensor_metadata_change(), "set_storage_offset is not allowed on Tensor created from .data or .detach()");
    storage_offset_ = storage_offset;
  }

  /**
   * Like set_sizes_and_strides but assumes contiguous strides.
   *
   * WARNING: This function does not check if the requested
   * sizes/strides are in bounds for the storage that is allocated;
   * this is the responsibility of the caller
   *
   * WARNING: It is NOT valid to call this method on a Variable.
   * See Note [We regret making Variable hold a Tensor]
   */
  void set_sizes_contiguous(IntArrayRef new_size) {
    AT_CHECK(allow_tensor_metadata_change(), "set_sizes_contiguous is not allowed on Tensor created from .data or .detach()");
    AT_ASSERT(!is_variable());  // TODO: remove this when Variable and Tensor are merged
    auto old_dim = sizes_.size();
    auto new_dim = new_size.size();

    sizes_.resize(new_dim);
    for (size_t dim = 0; dim < new_dim; ++dim) {
      sizes_[dim] = new_size[dim];
    }

    update_to_contiguous_strides(old_dim);
    refresh_numel();
  }

  /**
   * Set the sizes and strides of a tensor.
   *
   * WARNING: This function does not check if the requested
   * sizes/strides are in bounds for the storage that is allocated;
   * this is the responsibility of the caller
   *
   * WARNING: It is NOT valid to call this method on a Variable.
   * See Note [We regret making Variable hold a Tensor]
   */
  void set_sizes_and_strides(IntArrayRef new_size, IntArrayRef new_stride) {
    AT_CHECK(allow_tensor_metadata_change(), "set_sizes_and_strides is not allowed on Tensor created from .data or .detach()");
    AT_ASSERT(!is_variable());  // TODO: remove this when Variable and Tensor are merged
    AT_CHECK(
        new_size.size() == new_stride.size(),
        "dimensionality of sizes (",
        new_size.size(),
        ") must match dimensionality of strides (",
        new_stride.size(),
        ")");
    auto new_dim = new_size.size();

    sizes_.resize(new_dim);
    for (size_t dim = 0; dim < new_dim; ++dim) {
      sizes_[dim] = new_size[dim];
    }

    strides_.resize(new_dim);
    if (new_dim > 0) {
      for (size_t dim = new_dim - 1; ; dim--) {
        if (new_stride[dim] >= 0) {
          strides_[dim] = new_stride[dim];
        } else {
          // XXX: This behavior is surprising and may need to be removed to
          // support negative strides. Some pytorch functions rely on it:
          // for example, torch.cat (run TestTorch.test_cat_empty).
          if (dim == new_dim - 1) {
            strides_[dim] = 1;
          } else {
            // Keep stride monotonically increasing to match NumPy.
            strides_[dim] = std::max<int64_t>(sizes_[dim + 1], 1) * strides_[dim + 1];
          }
        }
        if (dim == 0) break;
      }
    }

    refresh_numel();
    refresh_contiguous();
  }

  /**
   * Return the size of a tensor at some dimension.
   */
  virtual int64_t size(int64_t d) const;

  /**
   * Return the stride of a tensor at some dimension.
   */
  virtual int64_t stride(int64_t d) const;

  /**
   * True if a tensor is a variable.  See Note [Tensor versus Variable in C++]
   */
  bool is_variable() const { return autograd_meta_ != nullptr; };

  /**
   * Set whether a tensor allows changes to its metadata (e.g. sizes / strides / storage / storage_offset).
   */
  virtual void set_allow_tensor_metadata_change(bool value) {
    allow_tensor_metadata_change_ = value;
  }

  /**
   * True if a tensor allows changes to its metadata (e.g. sizes / strides / storage / storage_offset).
   */
  virtual bool allow_tensor_metadata_change() const {
    return allow_tensor_metadata_change_;
  }

  /**
   * Set the pointer to autograd metadata.
   */
  void set_autograd_meta(std::unique_ptr<c10::AutogradMetaInterface> autograd_meta) {
    autograd_meta_ = std::move(autograd_meta);
  }

  /**
   * Return the pointer to autograd metadata.
   */
  c10::AutogradMetaInterface* autograd_meta() const {
    return autograd_meta_.get();
  }

  /**
   * Detach the autograd metadata unique_ptr from this tensor, and return it.
   */
  std::unique_ptr<c10::AutogradMetaInterface> detach_autograd_meta() {
    return std::move(autograd_meta_);
  }

  // NOTE: `shallow_copy_and_detach()` does not copy the following TensorImpl fields:
  // 1. the AutogradMeta pointer, because it is unique for each Variable.
  // 2. the version counter, because although it lives in TensorImpl, the version counter is managed
  // by autograd, and the call sites of `shallow_copy_and_detach()` (from autograd) should decide what
  // the version counter should be for each new TensorImpl. See NOTE [ Version Counter Sharing ] for details.
  //
  // NOTE: We don't set `allow_tensor_metadata_change_` to false here, because there are call sites
  // to this function that need to change the shallow copy's size or storage afterwards, and setting
  // `allow_tensor_metadata_change_` to false would prevent those changes from happening and is
  // undesirable.
  virtual c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach() const {
    AT_ASSERT(!is_variable());  // TODO: remove this when Variable and Tensor are merged
    auto impl = c10::make_intrusive<TensorImpl>(Storage(storage()), type_id());
    impl->set_sizes_and_strides(sizes(), strides());
    impl->storage_offset_ = storage_offset_;
    impl->is_wrapped_number_ = is_wrapped_number_;
    impl->reserved_ = reserved_;
    impl->refresh_numel();
    impl->refresh_contiguous();
    return impl;
  }

  void set_version_counter(
    const c10::VariableVersion& version_counter) noexcept {
    version_counter_ = version_counter;
  }

  const c10::VariableVersion& version_counter() const noexcept {
    return version_counter_;
  }

  void bump_version() noexcept {
    version_counter_.bump();
  }

  inline void set_pyobj(PyObject* pyobj) noexcept {
    pyobj_ = pyobj;
  }

  inline PyObject* pyobj() const noexcept {
    return pyobj_;
  }

 private:
  // See NOTE [c10::optional operator usage in CUDA]
  // We probably don't want to expose this publically until
  // the note is addressed.
  c10::optional<c10::Device> device_opt() const {
    return device_opt_;
  }

 public:

  /**
   * The device type of a Tensor, e.g., DeviceType::CPU or DeviceType::CUDA.
   */
  DeviceType device_type() const {
    AT_ASSERT(!is_variable());  // TODO: remove this when Variable and Tensor are merged
    AT_ASSERT(device_opt_.has_value());
    // See NOTE [c10::optional operator usage in CUDA]
    return (*device_opt_).type();
  }

  /**
   * The device of a Tensor; e.g., Device(kCUDA, 1) (the 1-index CUDA
   * device).
   */
  Device GetDevice() const {
    // See NOTE [c10::optional operator usage in CUDA]
    return *device_opt_;
  }

  /**
   * @brief Extends the outer-most dimension of this tensor by num elements,
   * preserving the existing data.
   *
   * The underlying data may be reallocated in order to accommodate the new
   * elements, in which case this tensors' capacity is grown at a factor of
   * growthPct. This ensures that Extend runs on an amortized O(1) time
   * complexity.
   *
   * This op is auto-asynchronous if the underlying device (CUDA) supports it.
   */
  void Extend(int64_t num, float growthPct) {
    AT_ASSERT(sizes_.size() >= 1u);
    AT_ASSERTM(num >= 0, "`num` must be non-negative for Extend");
    AT_ASSERTM(
        is_contiguous_,
        "Right now Extend is only supported for contiguous Tensor.");
    auto newDims = sizes_;
    newDims[0] += num;
    if (!storage_.data()) {
      Resize(newDims);
      return;
    }
    auto newNumel = std::accumulate(
        newDims.begin(),
        newDims.end(),
        static_cast<int64_t>(1),
        std::multiplies<int64_t>());
    if (newNumel * storage_.itemsize() <= storage_.capacity()) {
      sizes_ = newDims;
      numel_ = newNumel;
      return;
    }
    auto newCapacity = sizes_;
    newCapacity[0] = std::max<size_t>(
        newDims[0], std::ceil(sizes_[0] * (growthPct + 100) / 100));
    auto oldData = std::move(storage_.data_ptr());
    auto oldSize = numel_;
    auto oldDims = sizes_;
    Resize(newCapacity);
    auto* newData = raw_mutable_data(data_type_);
    if (data_type_.copy()) {
      AT_ASSERTM(
          device_type() == DeviceType::CPU,
          "non-POD types work only on CPU");
      data_type_.copy()(oldData.get(), newData, oldSize);
    } else {
      // The following copy uses the current (thread local) stream for copying
      // and also takes the GPU id from the device() field passed in.
      //
      // TODO: Potentially more enforcements are necessary to avoid accidental
      // switch to sync copy if the currently set device is wrong.
      //
      // Specifically, we might need to switch to a different context device
      // here explicitly to avoid relying on user synchronizing things
      // properly.
      CopyBytes(
          oldSize * itemsize(),
          oldData.get(),
          device(),
          newData,
          device(),
          true); // non-blocking
    }
    reserved_ = true;
    sizes_ = newDims;
    numel_ = newNumel;
  }

  /**
   * @brief Reserve space for the underlying tensor.
   *
   * This must be called after Resize(), since we only specify the first
   * dimension This does not copy over the old data to the newly allocated space
   */
  template <class T>
  void ReserveSpace(const T& outer_dim) {
    AT_ASSERTM(
        is_contiguous_,
        "Right now ReserveSpace is only supported for contiguous Tensor.");
    AT_ASSERTM(
        storage_.unique(), "Can't call ReserveSpace on shared storage.");
    auto newCapacity = sizes_;
    newCapacity[0] = outer_dim;
    auto newNumel = std::accumulate(
        newCapacity.begin(),
        newCapacity.end(),
        static_cast<int64_t>(1),
        std::multiplies<int64_t>());
    if (newNumel * storage_.itemsize() <= storage_.capacity()) {
      return;
    }
    // Old data is discarded
    storage_.data_ptr().clear();
    auto oldSize = numel_;
    auto oldDims = sizes_;
    Resize(newCapacity);
    // Allocate new memory but don't copy over the data
    raw_mutable_data(data_type_);
    sizes_ = oldDims;
    numel_ = oldSize;
    reserved_ = true;
  }

  /**
   * @brief Resizes a tensor.
   *
   * Resize takes in a vector of ints specifying the dimensions of the tensor.
   * You can pass in an empty vector to specify that it is a scalar (i.e.
   * containing one single item).
   *
   * The underlying storage may be deleted after calling Resize: if the new
   * shape leads to a different number of items in the tensor, the old memory
   * is deleted and new memory will be allocated next time you call
   * mutable_data(). However, if the shape is different but the total number of
   * items is the same, the underlying storage is kept.
   *
   * This method respects caffe2_keep_on_shrink.  Consult the internal logic
   * of this method to see exactly under what circumstances this flag matters.
   */
  template <typename... Ts>
  void Resize(Ts... dim_source) {
    bool size_changed = SetDims(dim_source...);
    if (size_changed) {
      // If needed, we will free the data. the next mutable_data() call
      // will create the data storage.
      bool reset_tensor = false;
      if (reserved_) {
        // If tensor is reserved then don't claim its memeory unless capacity()
        // is smaller than new size
        reset_tensor = storage_.capacity() < (storage_offset_ + numel_) * storage_.itemsize();
      } else {
        reset_tensor = storage_.capacity() <
                (storage_offset_ + numel_) * storage_.itemsize() ||
            !FLAGS_caffe2_keep_on_shrink ||
            storage_.capacity() -
                    (storage_offset_ + numel_) * storage_.itemsize() >
                static_cast<size_t>(FLAGS_caffe2_max_keep_on_shrink_memory);
      }

      if (reset_tensor && storage_initialized()) {
        FreeMemory();
      }
    }
  }

  /**
   * Resizes the tensor without touching underlying storage.
   * This requires the total size of the tensor to remains constant.
   */
  inline void Reshape(const std::vector<int64_t>& dims) {
    AT_ASSERTM(
        is_contiguous_,
        "Right now Reshape is only supported for contiguous Tensor.");
    int64_t new_size = 1;
    for (auto d : dims) {
      AT_ASSERT(d >= 0);
      new_size *= d;
    }
    AT_ASSERTM(
        new_size == numel_,
        "New size and old size are not equal. You cannot use Reshape, "
        "but should use Resize."
        // TODO(jiayq): remove the following warning after pending diffs
        // stabilize.
        " The old caffe2 mixes Reshape and Resize but this behavior has "
        "been changed. If you find this error, most likely you will need "
        "to change corresponding code from Reshape to Resize.");
    auto old_dim = sizes_.size();
    sizes_ = dims;
    update_to_contiguous_strides(old_dim);
  }

  /**
   * Release whatever memory the tensor was holding but keep size and type
   * information. Subsequent call to mutable_data will trigger new memory
   * allocation.
   */
  inline void FreeMemory() {
    // We'll detach from the old Storage and create a new one
    storage_ = Storage::create_legacy(storage_.device(), data_type_);
    storage_offset_ = 0;
  }

   /**
   * @brief Shares the data with another tensor.
   *
   * To share data between two tensors, the sizes of the two tensors must be
   * equal already. The reason we do not implicitly do a Resize to make the two
   * tensors have the same shape is that we want to allow tensors of different
   * shapes but the same number of items to still be able to share data. This
   * allows one to e.g. have a n-dimensional Tensor and a flattened version
   * sharing the same underlying storage.
   *
   * The source tensor should already have its data allocated.
   */
  // To be deprecated
  void ShareData(const TensorImpl& src) {
    // Right now, we are assuming the device_type are the same, since it is
    // inherently the same in the non-templatized code. We should probably add
    // an assert here which might affect perf a little bit.
    AT_ASSERTM(
        src.numel_ == numel_,
        "Size mismatch - did you call reshape before sharing the data?");
    // It is possible that the source tensor hasn't called mutable_data() yet,
    // in which case ShareData() doesn't make much sense since we don't really
    // know what to share yet.
    // TODO: Add the assert after all uninitialized states are eliminated
    // AT_ASSERTM(src.dtype_initialized(),
    //            "Source tensor don't have a data type (did you call mutable_data<T> on the tensor?)");
    if (!src.dtype_initialized()) {
      C10_LOG_EVERY_MS(WARNING, 1000) <<
                   "Source tensor don't have a data type (did you call mutable_data<T> on the tensor?)";
    }
    AT_ASSERTM(
        src.storage_initialized(),
        "Source tensor has no content and has size > 0");
    // Finally, do sharing.
    /* Since we create new Storage whenever we need to change data_type/capacity
     * this still keeps the original semantics
     */
    storage_ = src.storage();
    data_type_ = src.dtype();
    device_opt_ = src.device_opt();
    storage_offset_ = src.storage_offset();
  }

  void ShareExternalPointer(
      DataPtr&& data_ptr,
      const caffe2::TypeMeta& data_type,
      size_t capacity) {
    AT_ASSERTM(
        data_type.id() != caffe2::TypeIdentifier::uninitialized(),
        "To share with a raw external pointer you need to pass in an "
        "initialized data_type(TypeMeta).");
    if (!capacity) {
      capacity = numel_ * data_type.itemsize();
    }
    if (storage_.unique()) {
      storage_.UniqueStorageShareExternalPointer(
          std::move(data_ptr), data_type, capacity);
      data_type_ = data_type;
      device_opt_ = storage_.device();
      storage_offset_ = 0;
    } else {
      int64_t numel = capacity / data_type.itemsize();
      // Create a new Storage
      storage_ = Storage(
          data_type,
          numel,
          std::move(data_ptr),
          /*allocator=*/nullptr,
          /*resizable=*/false);
      data_type_ = data_type;
      device_opt_ = storage_.device();
      storage_offset_ = 0;
    }
  }

  /**
   * Returns a mutable raw pointer of the underlying storage. Since we will need
   * to know the type of the data for allocation, a TypeMeta object is passed in
   * to specify the necessary information. This is conceptually equivalent of
   * calling mutable_data<T>() where the TypeMeta parameter meta is derived from
   * the type T. This function differs from mutable_data<T>() in the sense that
   * the type T can be specified during runtime via the TypeMeta object.
   *
   * If the existing data does not match the desired type, it will be deleted
   * and a new storage will be created.
   */
  inline void* raw_mutable_data(const caffe2::TypeMeta& meta) {
    // For 0-size tensors it's fine to return any pointer (including nullptr)
    if (data_type_ == meta && storage_initialized()) {
      return static_cast<void*>(static_cast<char*>(storage_.data()) + storage_offset_ * meta.itemsize());
    } else {
      bool had_special_dtor = data_type_.placementDelete() != nullptr;
      storage_offset_ = 0;
      if (storage_.unique()) {
        storage_.set_dtype(meta);
      } else {
        if (data_type_ != meta) {
          storage_ = Storage::create_legacy(storage_.device(), meta);
        }
      }
      data_type_ = meta;
      // NB: device is not changed

      // We can reuse the existing buffer if the current data does not have
      // a special destructor and the new data doesn't have a special
      // constructor.
      if (numel_ == 0 ||
          (meta.placementNew() == nullptr && !had_special_dtor &&
           storage_.numel() >= numel_)) {
        AT_ASSERT(storage_offset_ == 0); // because we just reallocated
        return storage_.data();
      }
      const Allocator* allocator = storage_.allocator();
      // Storage might have nullptr allocator in rare cases, for example, if
      // an external memory segment has been wrapped with Tensor and we don't
      // know how to reallocate it. However, in order to preserve legacy C2
      // behavior, we allow reallocating the memory using default allocator.
      if (allocator == nullptr) {
        allocator = GetAllocator(storage_.device_type());
      }
      if (meta.placementNew()) {
        // For types that need placement new, we will call it, as well as
        // making sure that when the data is freed, it calls the right
        // destruction procedure.
        auto size = numel_;
        auto dtor = data_type_.placementDelete();
        auto data_ptr = allocator->allocate(numel_ * storage_.itemsize());
        storage_.set_data_ptr(PlacementDeleteContext::makeDataPtr(
            std::move(data_ptr), dtor, size, storage_.device()));
        data_type_.placementNew()(storage_.data(), numel_);
      } else {
        // For fundamental type, new and delete is easier.
        storage_.set_data_ptr(
            allocator->allocate(numel_ * storage_.itemsize()));
      }
      storage_.set_numel(numel_);
      AT_ASSERT(storage_offset_ == 0); // because we just reallocated
      device_opt_ = storage_.device();
      return storage_.data();
    }
  }

  /**
   * Returns a typed pointer of the underlying storage.
   *
   * For fundamental types, we reuse possible existing storage if there
   * is sufficient capacity.
   */
  template <typename T>
  inline T* mutable_data() {
    if (storage_initialized() && storage_.IsType<T>()) {
      return static_cast<T*>(storage_.data()) + storage_offset_;
    }
    // Check it here statically - otherwise TypeMeta would throw the runtime
    // error in attempt to invoke TypeMeta::ctor()
    static_assert(
        std::is_default_constructible<T>::value,
        "Tensor can't hold non-default-constructible types");
    return static_cast<T*>(raw_mutable_data(caffe2::TypeMeta::Make<T>()));
  }

  /**
   * True if a tensor is storage initialized.  A tensor may become
   * storage UNINITIALIZED after a Resize() or FreeMemory()
   */
  bool storage_initialized() const {
    AT_ASSERT(has_storage());
    return storage_.data() || numel_ == 0;
  }

  /**
   * True if a tensor is dtype initialized.  A tensor allocated with
   * Caffe2-style constructors is dtype uninitialized until the
   * first time mutable_data<T>() is called.
   */
  bool dtype_initialized() const noexcept {
    return data_type_ != caffe2::TypeMeta();
  }

  void set_storage(at::Storage storage) {
    AT_CHECK(allow_tensor_metadata_change(), "set_storage is not allowed on Tensor created from .data or .detach()");
    storage_ = std::move(storage);
    data_type_ = storage_.dtype();
    device_opt_ = storage_.device();
  }

private:

  // The Caffe2 Resize() method supports being called both as Resize({2,2}) as
  // well as variadic with Resize(2, 2).  These overloads provide all of the
  // supported calling configurations, while being overloads (and not templates)
  // so that implicit conversions still work.
  //
  // SetDims on ArrayRef is internally implemented as a template, so we can
  // handle both ArrayRefs of different types (there are some uses of
  // Resize in Caffe2 which pass in int, not int64_t.)

  template <
      typename T,
      typename = typename std::enable_if<std::is_integral<T>::value>::type>
  bool SetDimsTemplate(ArrayRef<T> src) {
    auto old_numel = numel_;
    auto old_dim = sizes_.size();
    sizes_.resize(src.size());
    int64_t new_numel = 1;
    for (size_t i = 0; i < src.size(); ++i) {
      new_numel *= src[i];
      sizes_[i] = src[i];
    }
    update_to_contiguous_strides(old_dim);
    numel_ = new_numel;
    return numel_ != old_numel;
  }

  bool SetDims(ArrayRef<int64_t> s) {
    return SetDimsTemplate(s);
  }

  bool SetDims(ArrayRef<int> s) {
    return SetDimsTemplate(s);
  }

  bool SetDims(ArrayRef<size_t> s) {
    return SetDimsTemplate(s);
  }

  bool SetDims() {
    return SetDims(IntArrayRef{});
  }

  bool SetDims(const int64_t d0) {
    return SetDims(IntArrayRef{d0});
  }

  bool SetDims(const int64_t d0, const int64_t d1) {
    return SetDims(IntArrayRef{d0, d1});
  }

  bool SetDims(const int64_t d0, const int64_t d1, const int64_t d2) {
    return SetDims(IntArrayRef{d0, d1, d2});
  }

  bool SetDims(const int64_t d0, const int64_t d1, const int64_t d2, const int64_t d3) {
    return SetDims(IntArrayRef{d0, d1, d2, d3});
  }

  inline void update_to_contiguous_strides(size_t old_dim) {
    strides_.resize(sizes_.size(), 0);
    if (dim() > 0) {
      int last_idx = dim() - 1;
      strides_[last_idx] = 1;
      for (auto i = last_idx - 1; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * std::max<int64_t>(sizes_[i + 1], 1);
      }
    }
    is_contiguous_ = true;
  }

  /**
   * Compute the number of elements based on the sizes of a tensor.
   */
  int64_t compute_numel() const {
    int64_t n = 1;
    for (auto s : sizes()) {
      n *= s;
    }
    return n;
  }

  /**
   * Compute whether or not a tensor is contiguous based on the sizes and
   * strides of a tensor.
   */
  bool compute_contiguous() const;

protected:
  /**
   * Recompute the cached numel of a tensor.  Call this if you modify sizes.
   */
  void refresh_numel() {
    AT_ASSERT(!is_variable());  // TODO: remove this when Variable and Tensor are merged
    numel_ = compute_numel();
  }

  /**
   * Recompute the cached contiguity of a tensor.  Call this if you modify sizes
   * or strides.
   */
  void refresh_contiguous() {
    AT_ASSERT(!is_variable());  // TODO: remove this when Variable and Tensor are merged
    is_contiguous_ = compute_contiguous();
  }

protected:
  Storage storage_;
  // This pointer points to an AutogradMeta struct that stores autograd-specific fields
  // (such as grad_ / grad_fn_ / grad_accumulator_).
  // This pointer always has unique ownership (meaning only one TensorImpl can own it
  // at a time).
  std::unique_ptr<c10::AutogradMetaInterface> autograd_meta_ = nullptr;

  c10::VariableVersion version_counter_;

  PyObject* pyobj_ = nullptr; // weak reference

  // We could save a word or two by combining the SmallVector structs,
  // since their size is redundant, and if we need to overflow the buffer space
  // we could keep the two pointers together. However, that would require
  // implementing another struct from scratch, so only do this if we're desperate.
  SmallVector<int64_t,5> sizes_;
  SmallVector<int64_t,5> strides_;

  int64_t storage_offset_ = 0;
  // If sizes and strides are empty, the numel is 1!!  However, most of the
  // time, we will immediately set sizes to {0} and reset numel to 0.
  // (Can't do that in the default initializers, because there's no way to
  // spell "allocate a one-element array" for strides_).
  int64_t numel_ = 1;

  // INVARIANT: When storage is non-null, this type meta must
  // agree with the type meta in storage
  caffe2::TypeMeta data_type_;

  // NOTE [c10::optional operator usage in CUDA]
  // Our optional definition doesn't compile in .cu file if `value()` or
  // `operator->` are used.  Instead, we always use `operator*`.
  // See https://github.com/pytorch/pytorch/issues/18496 for more info.
  // If this is too burdensome to maintain, we can just
  // manually implement this with an additional bool.

  // INVARIANT: When storage is non-null, this Device must
  // agree with the type meta in storage.
  c10::optional<c10::Device> device_opt_;

  // You get to have eight byte-size fields here, before you
  // should pack this into a bitfield.
  TensorTypeId type_id_;
  bool is_contiguous_ = true;
  bool is_wrapped_number_ = false;

  // Previously, if we change the tensor metadata (e.g. sizes / strides / storage / storage_offset)
  // of a derived tensor (i.e. tensors created from Python `tensor.data` or Python/C++ `tensor.detach()`),
  // those metadata in the original tensor will also be updated. However, the new behavior is that
  // those metadata changes to a derived tensor will not update the original tensor anymore, and we
  // need this flag to make such changes explicitly illegal, to prevent users from changing metadata of
  // the derived tensor and expecting the original tensor to also be updated.
  //
  // NOTE: For a full list of tensor metadata fields, please see `shallow_copy_and_detach()` in TensorImpl
  // and its subclasses to find which fields are copied by value.
  bool allow_tensor_metadata_change_ = true;

  // we decide to keep reserved_ and it will
  // live in Tensor after the split
  // The logic is that if Extend() or ReserveSpace() were ever called,
  // then subsequent Resize()s will not free up Storage.
  bool reserved_ = false;

};

// Note [TensorImpl size constraints]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Changed the size of TensorImpl?  If the size went down, good for
// you!  Adjust the documentation below and the expected size.
// Did it go up?  Read on...
//
// Struct size matters.  In some production systems at Facebook, we have
// 400M live tensors during a training run.  Do the math: every 64-bit
// word you add to Tensor is an extra 3.2 gigabytes in RAM.
//
// If you are a Facebook employee, you can check if the run in question
// has tipped you over the point using the command here:
// https://fburl.com/q5enpv98
//
// For reference, we OOMed at 160 bytes (20 words) per TensorImpl.
// This is not counting overhead from strides out-of-line allocation and
// StorageImpl space and this is from before we inlined sizes and strides
// directly into TensorImpl as SmallVectors.
//
// Our memory usage on 32-bit systems is suboptimal, but we're not checking
// for it at the moment (to help avoid rage inducing cycles when the
// 32-bit number is wrong).
//
// Current breakdown:
//
//    vtable pointer
//    strong refcount           TODO: pack these into one word
//    weak refcount
//    storage pointer
//    autograd metadata pointer
//    version counter (word 0)
//    version counter (word 1)
//    PyObject pointer
//    sizes SmallVector (begin)
//    sizes SmallVector (end)
//    sizes SmallVector (capacity)
//    sizes SmallVector (pre-allocated 0)
//    sizes SmallVector (pre-allocated 1)
//    sizes SmallVector (pre-allocated 2)
//    sizes SmallVector (pre-allocated 3)
//    sizes SmallVector (pre-allocated 4)
//    strides SmallVector (begin)
//    strides SmallVector (end)
//    strides SmallVector (capacity)
//    strides SmallVector (pre-allocated 0)
//    strides SmallVector (pre-allocated 1)
//    strides SmallVector (pre-allocated 2)
//    strides SmallVector (pre-allocated 3)
//    strides SmallVector (pre-allocated 4)
//    storage offset
//    numel
//    data type pointer
//    (optional) device
//    miscellaneous bitfield
//
static_assert(sizeof(void*) != sizeof(int64_t) || // if 64-bit...
              sizeof(TensorImpl) == sizeof(int64_t) * 29,
              "You changed the size of TensorImpl on 64-bit arch."
              "See Note [TensorImpl size constraints] on how to proceed.");

} // namespace c10
