#include <THD/base/data_channels/DataChannelTCP.hpp>

#include <sys/poll.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>

namespace thd {
namespace {

inline uint32_t log2ceil(uint32_t value) {
  uint32_t dim = 0;
#if defined(__GNUC__)
  if (value <= 1)
    return 0;
  dim = 32 - __builtin_clz(value - 1);
#else
  for (uint32_t size = 1; size < value; ++dim, size <<= 1) /* empty */
    ;
#endif // defined(__GNUC__)
  return dim;
}

// Finds nearest power-of-two less than or equal to `value`.
template <typename T>
inline uint64_t pow2(T value) {
  uint64_t pof2 = 1;
  while (pof2 <= value) {
    pof2 <<= 1;
  }
  pof2 >>= 1;
  return pof2;
}

} // namespace

DataChannelTCP::RequestTCP::RequestTCP(QueueWorker::Request&& request)
    : _request(std::move(request)) {}

DataChannelTCP::RequestTCP::~RequestTCP() {}

bool DataChannelTCP::RequestTCP::isCompleted() {
  return _request.isCompleted();
}

void DataChannelTCP::RequestTCP::wait() {
  _request.wait();
}

DataChannelTCP::DataChannelTCP(InitMethod::Config config)
    : DataChannelTCP(config, -1) {}

DataChannelTCP::DataChannelTCP(InitMethod::Config config, int timeout)
    : _socket(-1),
      _port(0),
      _timeout(timeout),
      _processes(config.world_size),
      _poll_events(nullptr) {
  _rank = config.rank;

  if (_rank == 0) { // MASTER
    _socket = config.master.listen_socket;
    _port = config.master.listen_port;

    _processes[0] = {
        .rank = 0,
        .address = "",
        .port = 0,
        .socket = -1,
    };
  } else { // WORKER
    // add master
    _processes[0] = {
        .rank = 0,
        .address = config.worker.master_addr,
        .port = config.worker.master_port,
        .socket = -1,
    };
  }
}

DataChannelTCP::~DataChannelTCP() {
  if (_socket != -1)
    ::close(_socket);

  for (const auto& process : _processes) {
    if ((process.rank != _rank) && (process.socket != -1))
      ::close(process.socket);
  }
}

void DataChannelTCP::destroy() {}

bool DataChannelTCP::initWorker() {
  auto& master = _processes[0];
  master.socket = connect(master.address, master.port);

  std::tie(_socket, _port) = listen();

  send_value<rank_type>(master.socket, _rank, true);
  send_value<port_type>(master.socket, _port); // send listening port to master

  // get all metadata of other processes in network
  for (size_t i = 1; i < _processes.size(); ++i) {
    rank_type p_rank = recv_value<rank_type>(master.socket);
    port_type p_port = recv_value<port_type>(master.socket);
    std::string p_address = recv_string(master.socket);

    _processes[p_rank] = {
        .rank = p_rank,
        .address = p_address,
        .port = p_port,
        .socket = -1,
    };
  }

  /*
   * Firstly we are connecting to workers with rank lower than our rank,
   * then we accepting connections from other wokers with higher rank.
   *
   * This prevents from deadlocks where everyone is accepting or everyone is
   * trying to connect.
   */

  for (rank_type r = 1; r < _rank; ++r) {
    auto& process = _processes[r];
    process.socket = connect(process.address, process.port);

    // send rank to tell to the accepting process who we are
    send_value<rank_type>(process.socket, _rank);
  }

  for (rank_type i = _rank + 1; i < _processes.size(); ++i) {
    int socket;
    std::tie(socket, std::ignore) = accept(_socket, _timeout);

    // get rank of process we have just accepted
    rank_type p_rank = recv_value<rank_type>(socket);
    _processes[p_rank].socket = socket;
  }

  // close socket for listening, we will not use it anymore
  ::close(_socket);
  _socket = -1;

  return true;
}

bool DataChannelTCP::initMaster() {
  // wait for all workers to connect
  for (size_t i = 1; i < _processes.size(); ++i) {
    std::string p_address;
    int p_socket;
    std::tie(p_socket, p_address) = accept(_socket, _timeout);

    rank_type p_rank = recv_value<rank_type>(p_socket);
    port_type p_port = recv_value<port_type>(p_socket);

    if (p_rank >= _processes.size()) {
      throw std::out_of_range(
          "worker's rank(" + std::to_string(p_rank) +
          ") is out"
          "of range: [0, " +
          std::to_string(_processes.size() - 1) + "]");
    }

    if (_processes[p_rank].rank == p_rank) {
      throw std::logic_error(
          "two processes (" + _processes[p_rank].address + ", " + p_address +
          ") "
          "reported a rank of " +
          std::to_string(p_rank));
    }

    _processes[p_rank] = {
        .rank = p_rank,
        .address = p_address,
        .port = p_port,
        .socket = p_socket,
    };
  }

  // send informations about processes to all workers
  for (const auto& worker : _processes) {
    if (worker.rank == 0)
      continue;

    for (auto& process : _processes) {
      if (process.rank == 0)
        continue;

      send_value<rank_type>(worker.socket, process.rank, true);
      send_value<port_type>(worker.socket, process.port, true);
      send_string(worker.socket, process.address);
    }
  }

  // close socket for listening, we will not use it anymore
  ::close(_socket);
  _socket = -1;

  return true;
}

bool DataChannelTCP::init() {
  bool ok = (_rank == 0 ? initMaster() : initWorker());
  if (ok) {
    std::vector<rank_type> ranks;
    ranks.reserve(_processes.size());
    for (rank_type rank = 0; rank < _processes.size(); ++rank)
      ranks.push_back(rank);

    _groups.insert(
        {THDGroupWORLD, DataChannel::Group(ranks, _processes.size() - 1)});
  }

  return ok;
}

rank_type DataChannelTCP::getRank() {
  return _rank;
}

rank_type DataChannelTCP::getNumProcesses() {
  return _processes.size();
}

void DataChannelTCP::allGather(
    std::vector<at::Tensor>& output,
    at::Tensor& input,
    THDGroup group_id) {
  /*
   * Allgather algorithm is simple ring algorithm. This algorithm perfroms
   * well on large data (> 512 KB) and generalize well on large group of nodes.
   * More about efficiency can be found here:
   *   > http://www.mcs.anl.gov/~thakur/papers/ijhpca-coll.pdf (section 4.1)
   *
   * TODO: implement Bruck / recursive doubling algorithms to make allGather
   * efficient also for small data (< 512 KB).
   */

  std::lock_guard<std::mutex> lock(_mutex);

  const auto& group = _groups.at(group_id);
  rank_type group_rank;
  bool exists;
  std::tie(group_rank, exists) = group.getGroupRank(_rank);
  if (!exists)
    return;

  if (output.size() != group.size())
    throw std::logic_error(
        "allGather: number of output tensors and group size does not match");

  for (auto out_tensor : output)
    assertSameSizeAndType(out_tensor, input, "allGather");

  rank_type left = (group.size() + group_rank - 1) % group.size();
  rank_type right = (group_rank + 1) % group.size();

  memcpy(
      output[group_rank].data_ptr(),
      input.data_ptr(),
      input.element_size() * input.numel());

  auto j = group_rank, jnext = left;
  for (rank_type i = 0; i < group.size(); ++i) {
    req_ptr send_request{isend((output[j]), group.mustGetGlobalRank(right))};
    receive((output[jnext]), group.mustGetGlobalRank(left));
    send_request->wait();

    j = jnext;
    jnext = (group.size() + jnext - 1) % group.size();
  }
}

void DataChannelTCP::gather(
    std::vector<at::Tensor>& output,
    at::Tensor& input,
    rank_type dst_rank,
    THDGroup group_id) {
  std::lock_guard<std::mutex> lock(_mutex);

  const auto& group = _groups.at(group_id);
  bool exists;

  std::tie(std::ignore, exists) = group.getGroupRank(_rank);
  if (!exists)
    return;

  // assert if dst_rank exists in group
  group.mustGetGroupRank(dst_rank);
  if (_rank != dst_rank) {
    send(input, dst_rank);
  } else {
    if (output.size() != group.size())
      throw std::logic_error(
          "gather: number of output tensors and group size does not match");

    for (auto out_tensor : output)
      assertSameSizeAndType(out_tensor, input, "gather");

    for (rank_type i = 0; i < group.size(); ++i) {
      auto global_rank = group.mustGetGlobalRank(i);
      if (_rank != global_rank) {
        receive((output.at(i)), global_rank);
      } else {
        memcpy(
            output.at(i).data_ptr(),
            input.data_ptr(),
            input.numel() * input.element_size());
      }
    }
  }
}

void DataChannelTCP::scatter(
    std::vector<at::Tensor>& input,
    at::Tensor& output,
    rank_type src_rank,
    THDGroup group_id) {
  std::lock_guard<std::mutex> lock(_mutex);

  const auto& group = _groups.at(group_id);
  bool exists;

  std::tie(std::ignore, exists) = group.getGroupRank(_rank);
  if (!exists)
    return;

  // assert if src_rank exists in group
  group.mustGetGroupRank(src_rank);
  if (_rank != src_rank) {
    receive(output, src_rank);
  } else {
    if (input.size() != group.size())
      throw std::logic_error(
          "scatter: number of input tensors and group size does not match");

    for (auto in_tensor : input)
      assertSameSizeAndType(in_tensor, output, "scatter");

    for (rank_type i = 0; i < group.size(); ++i) {
      auto global_rank = group.mustGetGlobalRank(i);
      if (_rank != global_rank) {
        send((input.at(i)), global_rank);
      } else {
        memcpy(
            output.data_ptr(),
            input.at(i).data_ptr(),
            output.numel() * output.element_size());
      }
    }
  }
}

void DataChannelTCP::allReduce(
    at::Tensor& data,
    THDReduceOp operation,
    THDGroup group_id) {
  /*
   * Allreduce implementation is recursive doubling algorithm. It is good
   * algorithm for small sizes of message but other (theoratically better)
   * implementations could not be addapted because of non-commutative
   * operations on tensors (operation cannot be commutative because this could
   * introduce different numerical errors on different workers).
   *
   * More about efficiency can be found here:
   *   > http://www.mcs.anl.gov/~thakur/papers/ijhpca-coll.pdf (section 4.5)
   *
   * Implementation is based on:
   *   > https://github.com/pmodels/mpich/blob/master/src/mpi/coll/allreduce.c
   */

  std::lock_guard<std::mutex> lock(_mutex);

  const auto& group = _groups.at(group_id);
  rank_type group_rank;
  bool exists;

  std::tie(group_rank, exists) = group.getGroupRank(_rank);
  if (!exists)
    return;

  uint64_t tensor_bytes = data.element_size() * data.numel();
  auto tmp_tensor = data.clone();

  auto pof2 = pow2(group.size());
  int rem = group.size() - pof2;
  int newrank = 0;

  if (group_rank < 2 * rem) {
    if (group_rank % 2 == 0) {
      send(data, group.mustGetGlobalRank(group_rank + 1));
      newrank = -1;
    } else {
      receive(tmp_tensor, group.mustGetGlobalRank(group_rank - 1));
      _reduce(data, tmp_tensor, operation);
      newrank = group_rank / 2;
    }
  } else {
    newrank = group_rank - rem;
  }

  if (newrank != -1) {
    int mask = 0x1;
    while (mask < pof2) {
      int newdst = newrank ^ mask;
      int dst = (newdst < rem) ? (newdst * 2 + 1) : (newdst + rem);

      auto dst_global_rank = group.mustGetGlobalRank(dst);
      req_ptr send_request{isend(data, dst_global_rank)};
      receive(tmp_tensor, dst_global_rank);
      send_request->wait();

      if (dst < group_rank) {
        _reduce(data, tmp_tensor, operation);
      } else {
        _reduce(tmp_tensor, data, operation);
        std::memcpy(data.data_ptr(), tmp_tensor.data_ptr(), tensor_bytes);
      }

      mask <<= 1;
    }
  }

  if (group_rank < 2 * rem) {
    if (group_rank % 2) {
      send(data, group.mustGetGlobalRank(group_rank - 1));
    } else {
      receive(data, group.mustGetGlobalRank(group_rank + 1));
    }
  }
}

void DataChannelTCP::reduce(
    at::Tensor& data,
    THDReduceOp operation,
    rank_type dst_rank,
    THDGroup group_id) {
  /*
   * Idea of this algorithm is similar to broadcast but with reversed
   * order and direction of communication.
   */

  std::lock_guard<std::mutex> lock(_mutex);

  const auto& group = _groups.at(group_id);
  rank_type group_rank;
  bool exists;

  std::tie(group_rank, exists) = group.getGroupRank(_rank);
  if (!exists)
    return;

  auto group_dst_rank = group.mustGetGroupRank(dst_rank);
  int dim = log2ceil(group.size());
  rank_type virtual_rank =
      (group_rank + group.size() - group_dst_rank) % group.size();
  int64_t mask = 0;
  auto result_tensor = data.clone();

  for (int k = 0; k <= dim - 1; mask ^= (1 << k), ++k) {
    if ((virtual_rank & mask) == 0) {
      rank_type partner =
          virtual_rank ^ (1 << k); // partner has opposite bit `k`
      if (partner >= group.size())
        continue;

      partner =
          group.mustGetGlobalRank((partner + group_dst_rank) % group.size());
      if ((virtual_rank & (1 << k)) != 0) {
        send(result_tensor, partner);
      } else {
        receive(data, partner);
        _reduce(result_tensor, data, operation);
      }
    }
  }

  if (_rank == dst_rank)
    std::memcpy(
        data.data_ptr(),
        result_tensor.data_ptr(),
        data.element_size() * data.numel());
}

void DataChannelTCP::broadcast(
    at::Tensor& data,
    rank_type src_rank,
    THDGroup group_id) {
  /*
   * General idea of this algorithm is to send data in `d` dimensional
   * hypercube where vertices are nodes (processes) and edges are
   * network connections which can be used to transfer data.
   *
   * Since hypercube algorithm works for case when broadcasting rank is 0
   * we have to create `virtual_rank` which converts regular ranks to
   * virtual ones where `virtual_rank` for `src_rank` is 0.
   */

  std::lock_guard<std::mutex> lock(_mutex);

  const auto& group = _groups.at(group_id);
  rank_type group_rank;
  bool exists;

  std::tie(group_rank, exists) = group.getGroupRank(_rank);
  if (!exists)
    return;

  auto group_src_rank = group.mustGetGroupRank(src_rank);
  int dim = log2ceil(group.size());
  rank_type virtual_rank =
      (group_rank + group.size() - group_src_rank) % group.size();
  int64_t mask = (1 << dim) - 1;

  for (int k = dim - 1; k >= 0; --k) {
    mask ^= (1 << k); // clear bit `k`
    if ((virtual_rank & mask) == 0) {
      rank_type partner =
          virtual_rank ^ (1 << k); // partner has opposite bit `k`
      if (partner >= group.size())
        continue;

      partner =
          group.mustGetGlobalRank((partner + group_src_rank) % group.size());
      if ((virtual_rank & (1 << k)) == 0) {
        send(data, partner);
      } else {
        receive(data, partner);
      }
    }
  }
}

void DataChannelTCP::send(Scalar& data, rank_type dst_rank) {
  auto request = _send_worker.push(
      [this, &data, dst_rank] { this->_send(data, dst_rank); });
  request.wait();
}

void DataChannelTCP::send(at::Tensor& data, rank_type dst_rank) {
  auto request = _send_worker.push(
      [this, &data, dst_rank] { this->_send(data, dst_rank); });
  request.wait();
}

void DataChannelTCP::receive(Scalar& data, rank_type src_rank) {
  auto request = _receive_worker.push(
      [this, &data, src_rank] { this->_receive(data, src_rank); });
  request.wait();
}

rank_type DataChannelTCP::receive(at::Tensor& data) {
  rank_type sender;
  auto request = _receive_worker.push([this, &data, &sender] {
    if (!this->_poll_events) {
      // cache poll events array, it will be reused in another `receive` calls
      this->_poll_events.reset(new struct pollfd[this->_processes.size()]);
      for (size_t rank = 0; rank < this->_processes.size(); ++rank) {
        this->_poll_events[rank] = {.fd = this->_processes[rank].socket,
                                    .events = POLLIN};
      }
    }

    // cleanup
    for (size_t rank = 0; rank < this->_processes.size(); ++rank) {
      this->_poll_events[rank].revents = 0;
    }

    SYSCHECK(::poll(
        this->_poll_events.get(),
        this->_processes.size(),
        -1)) // infinite timeout
    for (size_t rank = 0; rank < this->_processes.size(); ++rank) {
      if (this->_poll_events[rank].revents == 0)
        continue;

      if (this->_poll_events[rank].revents ^ POLLIN)
        throw std::system_error(ECONNABORTED, std::system_category());

      this->_receive(data, rank);
      sender = rank;
      break;
    }
  });

  request.wait();
  return sender;
}

void DataChannelTCP::receive(at::Tensor& data, rank_type src_rank) {
  auto request = _receive_worker.push(
      [this, &data, src_rank] { this->_receive(data, src_rank); });
  request.wait();
}

DataChannelTCP::RequestTCP* DataChannelTCP::isend(
    at::Tensor& data,
    rank_type dst_rank) {
  auto request = _send_worker.push(
      [this, data, dst_rank] { this->_send(data, dst_rank); });
  return new DataChannelTCP::RequestTCP(std::move(request));
}

DataChannelTCP::RequestTCP* DataChannelTCP::ireceive(
    at::Tensor& data,
    rank_type src_rank) {
  auto request = _receive_worker.push(
      [this, data, src_rank] { this->_receive(data, src_rank); });
  return new DataChannelTCP::RequestTCP(std::move(request));
}

void DataChannelTCP::barrier(THDGroup group_id) {
  /*
   * Barrier is implementation of Bruck algorithm. All processes send to
   * other processes with rank (i + 2^k) and recv from process with rank (i -
   * 2^k) with wrap-around. Since we cannot do recv and send at the same time we
   * do recv asynchronously (thread), send byte and then wait for recv to
   * complete.
   */

  std::lock_guard<std::mutex> lock(_mutex);

  const auto& group = _groups.at(group_id);
  rank_type group_rank;
  bool exists;

  std::tie(group_rank, exists) = group.getGroupRank(_rank);
  if (!exists)
    return;

  std::uint8_t byte = 1;
  for (rank_type distance = 1; distance < group.size(); distance <<= 1) {
    rank_type recv_partner =
        (group_rank + group.size() - distance) % group.size();
    const auto& recv_process =
        _processes.at(group.mustGetGlobalRank(recv_partner));
    auto recv_request = _receive_worker.push([&recv_process, &byte] {
      recv_bytes<std::uint8_t>(recv_process.socket, &byte, 1);
    });

    rank_type send_partner = (group_rank + distance) % group.size();
    const auto& send_process =
        _processes.at(group.mustGetGlobalRank(send_partner));
    auto send_request = _send_worker.push([&send_process, &byte] {
      send_bytes<std::uint8_t>(send_process.socket, &byte, 1);
    });

    send_request.wait();
    recv_request.wait();
  }
}

THDGroup DataChannelTCP::newGroup(const std::vector<rank_type>& ranks) {
  auto new_group = DataChannel::Group(ranks, _processes.size() - 1);
  THDGroup new_group_id = static_cast<THDGroup>(_groups.size());

  _groups.insert({new_group_id, new_group});
  return new_group_id;
}

void DataChannelTCP::_send(const Scalar& data, rank_type dst_rank) {
  /*
   * We have to check if dst_rank is positive to properly use `.at` function in
   * vector. Not checking that can result in int overflow and strange errors.
   */

  const auto& process_dst = _processes.at(dst_rank);
  if (process_dst.rank == _rank)
    throw std::logic_error("cannot send scalar to process with same rank");

  // send size of scalar in bytes
  uint64_t scalar_bytes = data.elementSize();
  send_bytes<uint64_t>(process_dst.socket, &scalar_bytes, 1, true);

  // send data (bytes)
  send_bytes<std::uint8_t>(
      process_dst.socket,
      reinterpret_cast<const std::uint8_t*>(data.data()),
      scalar_bytes);
}

void DataChannelTCP::_send(const at::Tensor& data, rank_type dst_rank) {
  /*
   * We have to check if dst_rank is positive to properly use `.at` function in
   * vector. Not checking that can result in int overflow and strange errors.
   */

  const auto& process_dst = _processes.at(dst_rank);
  if (process_dst.rank == _rank)
    throw std::logic_error("cannot send tensor to process with same rank");

  if (!data.is_contiguous())
    throw std::logic_error("tensor to send is not contiguous");

  // send size of tensor data in bytes
  uint64_t tensor_bytes = data.element_size() * data.numel();
  send_bytes<uint64_t>(process_dst.socket, &tensor_bytes, 1, true);

  // send data (bytes)
  send_bytes<std::uint8_t>(
      process_dst.socket,
      reinterpret_cast<const std::uint8_t*>(data.data_ptr()),
      tensor_bytes);
}

void DataChannelTCP::_receive(Scalar& data, rank_type src_rank) {
  /*
   * We have to check if src_rank is positive to properly use `.at` function in
   * vector. Not checking that can result in int overflow and strange errors.
   */

  const auto& process_src = _processes.at(src_rank);
  if (process_src.rank == _rank)
    throw std::logic_error("cannot receive scalar from process with same rank");

  // get size of scalar in bytes
  uint64_t scalar_bytes;
  recv_bytes<uint64_t>(process_src.socket, &scalar_bytes, 1);

  uint64_t actual_scalar_bytes = data.elementSize();
  if (actual_scalar_bytes == scalar_bytes) {
    recv_bytes<std::uint8_t>(
        process_src.socket,
        reinterpret_cast<std::uint8_t*>(data.data()),
        scalar_bytes);
  } else {
    // remove invalid data from recv buffer
    std::unique_ptr<std::uint8_t[]> bytes(new std::uint8_t[scalar_bytes]);
    recv_bytes<std::uint8_t>(process_src.socket, bytes.get(), scalar_bytes);
    throw std::logic_error("scalar sizes do not match");
  }
}

void DataChannelTCP::_receive(const at::Tensor& data, rank_type src_rank) {
  /*
   * We have to check if src_rank is positive to properly use `.at` function in
   * vector. Not checking that can result in int overflow and strange errors.
   */

  const auto& process_src = _processes.at(src_rank);
  if (process_src.rank == _rank)
    throw std::logic_error("cannot receive tensor from process with same rank");

  if (!data.is_contiguous())
    throw std::logic_error("tensor to receive is not contiguous");

  // get size of tensor data in bytes
  uint64_t tensor_bytes;
  recv_bytes<uint64_t>(process_src.socket, &tensor_bytes, 1);

  uint64_t actual_tensor_bytes =
      data.element_size() * data.numel();
  if (actual_tensor_bytes == tensor_bytes) {
    recv_bytes<std::uint8_t>(
        process_src.socket,
        reinterpret_cast<std::uint8_t*>(data.data_ptr()),
        tensor_bytes);
  } else {
    // remove invalid data from recv buffer
    std::unique_ptr<std::uint8_t[]> bytes(new std::uint8_t[tensor_bytes]);
    recv_bytes<std::uint8_t>(process_src.socket, bytes.get(), tensor_bytes);
    throw std::logic_error("tensor sizes do not match");
  }
}

void DataChannelTCP::_reduce(
    at::Tensor& result,
    at::Tensor& data,
    THDReduceOp operation) const {
  assertSameSizeAndType(result, data, "reduce");

  if (operation == THDReduceOp::THDReduceMIN) {
    at::min_out(result, result, data);
  } else if (operation == THDReduceOp::THDReduceMAX) {
    at::max_out(result, result, data);
  } else if (operation == THDReduceOp::THDReduceSUM) {
    result.add_(data);
  } else if (operation == THDReduceOp::THDReducePRODUCT) {
    result.mul_(data);
  } else {
    throw std::logic_error("unsupported reduce operation");
  }
}

void DataChannelTCP::allReduce(
    std::vector<at::Tensor>& data,
    THDReduceOp operation,
    THDGroup groupId) {
  throw std::runtime_error(
      "DataChannelTCP does not support mult-GPU cross "
      "node allreduce");
}

void DataChannelTCP::allGather(
    std::vector<at::Tensor>& output,
    std::vector<at::Tensor>& input,
    THDGroup groupId) {
  throw std::runtime_error(
      "DataChannelTCP does not support mult-GPU cross "
      "node allgather");
}

void DataChannelTCP::reduce(
    std::vector<at::Tensor>& data,
    THDReduceOp operation,
    rank_type dstRank,
    THDGroup groupId) {
  throw std::runtime_error(
      "DataChannelTCP does not support mult-GPU cross "
      "node reduce");
}

void DataChannelTCP::broadcast(
    std::vector<at::Tensor>& data,
    rank_type srcRank,
    THDGroup groupId) {
  throw std::runtime_error(
      "DataChannelTCP does not support mult-GPU cross "
      "node broadcast");
}

void DataChannelTCP::clearGroupCache(THDGroup group_id) {
  throw std::runtime_error(
      "DataChannelTCP does not support clear "
      "group cache");
}

} // namespace thd
