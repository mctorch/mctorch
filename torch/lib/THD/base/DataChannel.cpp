#include <THD/base/DataChannel.hpp>
#ifdef WITH_GLOO
#include <THD/base/data_channels/DataChannelGloo.hpp>
#endif // WITH_GLOO
#ifdef WITH_MPI
#include <THD/base/data_channels/DataChannelMPI.hpp>
#endif // WITH_MPI
#if defined(USE_CUDA) && defined(USE_DISTRIBUTED_NCCL)
#include <THD/base/data_channels/DataChannelNccl.hpp>
#endif // USE_DISTRIBUTED_NCCL
#include <THD/base/data_channels/DataChannelTCP.hpp>

#include <algorithm>
#include <stdexcept>
#include <tuple>

namespace thd {

#define GET_CONFIG getInitConfig(init_method, world_size, group_name, rank)
DataChannel* DataChannel::newChannel(
    THDChannelType type,
    std::string init_method,
    int world_size,
    std::string group_name,
    int rank) {
  switch (type) {
    case THDChannelTCP:
      return new DataChannelTCP(GET_CONFIG);

    case THDChannelMPI:
#ifdef WITH_MPI
      return new DataChannelMPI();
#endif // WITH_MPI
      throw std::runtime_error(
          "the MPI backend is not available; "
          "try to recompile the THD package with MPI support");

    case THDChannelGloo:
#ifdef WITH_GLOO
      return new DataChannelGloo(GET_CONFIG);
#endif // WITH_GLOO
      throw std::runtime_error(
          "the Gloo backend is not available; "
          "try to recompile the THD package with Gloo support");

    case THDChannelNccl:
#if defined(USE_CUDA) && defined(USE_DISTRIBUTED_NCCL)
      return new DataChannelNccl(GET_CONFIG);
#endif
      throw std::runtime_error(
          "the distributed NCCL backend is not available; "
          "try to recompile the THD package with CUDA and NCCL 2+ support");

    default:
      throw std::runtime_error("unsupported data channel type");
  }
}
#undef GET_CONFIG

DataChannel::Group::Group() {}

DataChannel::Group::Group(std::vector<rank_type> ranks, rank_type max_rank) {
  if (ranks.size() == 0)
    throw std::logic_error("cannot create empty group");

  sort(ranks.begin(), ranks.end());
  if (ranks.back() > max_rank) {
    throw std::out_of_range(
        "array of ranks contains invalid rank, "
        "all ranks should be in range: [0, " +
        std::to_string(max_rank) + "]");
  }

  _new2old.reserve(ranks.size());
  for (size_t i = 0; i < ranks.size(); ++i) {
    _new2old.push_back(ranks[i]);
    _old2new.insert({ranks[i], i});
  }
}

DataChannel::Group::~Group() {}

auto DataChannel::Group::size() const -> rank_type {
  return static_cast<rank_type>(_new2old.size());
}

auto DataChannel::Group::mustGetGroupRank(rank_type global_rank) const
    -> rank_type {
  rank_type group_rank;
  bool exists;
  std::tie(group_rank, exists) = getGroupRank(global_rank);

  if (!exists) {
    throw std::logic_error(
        "rank(" + std::to_string(global_rank) + ") is not member of group");
  }

  return group_rank;
}

auto DataChannel::Group::getGroupRank(rank_type global_rank) const
    -> std::pair<rank_type, bool> {
  auto global_rank_it = _old2new.find(global_rank); // O(1) operation
  if (global_rank_it != _old2new.end())
    return std::make_pair(global_rank_it->second, true);

  return std::make_pair(0, false);
}

auto DataChannel::Group::mustGetGlobalRank(rank_type group_rank) const
    -> rank_type {
  rank_type global_rank;
  bool exists;
  std::tie(global_rank, exists) = getGlobalRank(group_rank);

  if (!exists) {
    throw std::logic_error(
        "group rank is invalid, rank should be in "
        "range: [0, " +
        std::to_string(_new2old.size() - 1) + "]");
  }

  return global_rank;
}

auto DataChannel::Group::getGlobalRank(rank_type group_rank) const
    -> std::pair<rank_type, bool> {
  if (group_rank >= _new2old.size())
    return std::make_pair(0, false);

  return std::make_pair(_new2old[group_rank], true);
}

} // namespace thd
