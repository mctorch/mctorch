#include <THD/base/init_methods/InitMethodUtils.hpp>

#include <ifaddrs.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <unistd.h>

#include <tuple>

namespace thd {

namespace {

void sendPeerName(int socket) {
  struct sockaddr_storage master_addr;
  socklen_t master_addr_len = sizeof(master_addr);
  SYSCHECK(getpeername(
      socket,
      reinterpret_cast<struct sockaddr*>(&master_addr),
      &master_addr_len));

  std::string addr_str =
      sockaddrToString(reinterpret_cast<struct sockaddr*>(&master_addr));
  send_string(socket, addr_str);
}

} // namespace

std::vector<std::string> getInterfaceAddresses() {
  struct ifaddrs* ifa;
  SYSCHECK(getifaddrs(&ifa));
  ResourceGuard ifaddrs_guard([ifa]() { ::freeifaddrs(ifa); });

  std::vector<std::string> addresses;

  while (ifa != nullptr) {
    struct sockaddr* addr = ifa->ifa_addr;
    if (addr) {
      bool is_loopback = ifa->ifa_flags & IFF_LOOPBACK;
      bool is_ip = addr->sa_family == AF_INET || addr->sa_family == AF_INET6;
      if (is_ip && !is_loopback) {
        addresses.push_back(sockaddrToString(addr));
      }
    }
    ifa = ifa->ifa_next;
  }

  return addresses;
}

std::string discoverWorkers(int listen_socket, rank_type world_size) {
  // accept connections from workers so they can know our address
  std::vector<int> sockets(world_size - 1);
  for (rank_type i = 0; i < world_size - 1; ++i) {
    std::tie(sockets[i], std::ignore) = accept(listen_socket);
  }

  std::string public_addr;
  for (auto socket : sockets) {
    sendPeerName(socket);
    public_addr = recv_string(socket);
    ::close(socket);
  }
  return public_addr;
}

std::pair<std::string, std::string> discoverMaster(
    std::vector<std::string> addresses,
    port_type port) {
  // try to connect to address via any of the addresses
  std::string master_address = "";
  int socket;
  for (const auto& address : addresses) {
    try {
      socket = connect(address, port, true, 2000);
      master_address = address;
      break;
    } catch (...) {
    } // when connection fails just try different address
  }

  if (master_address == "") {
    throw std::runtime_error(
        "could not establish connection with other processes");
  }
  ResourceGuard socket_guard([socket]() { ::close(socket); });
  sendPeerName(socket);
  std::string my_address = recv_string(socket);

  return std::make_pair(master_address, my_address);
}

rank_type getRank(
    const std::vector<int>& ranks,
    int assigned_rank,
    size_t order) {
  if (assigned_rank >= 0) {
    return assigned_rank;
  } else {
    std::vector<bool> taken_ranks(ranks.size());
    for (auto rank : ranks) {
      if (rank >= 0)
        taken_ranks[rank] = true;
    }

    auto unassigned = std::count(ranks.begin(), ranks.begin() + order, -1) + 1;
    rank_type rank = 0;
    while (true) {
      if (!taken_ranks[rank])
        unassigned--;
      if (unassigned == 0)
        break;
      rank++;
    }

    return rank;
  }
}
} // namespace thd
