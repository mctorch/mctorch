#include <c10d/Utils.hpp>

#include <netdb.h>
#include <sys/poll.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

namespace c10d {
namespace tcputil {

namespace {

constexpr int LISTEN_QUEUE_SIZE = 64;

void setSocketNoDelay(int socket) {
  int flag = 1;
  socklen_t optlen = sizeof(flag);
  SYSCHECK_ERR_RETURN_NEG1(setsockopt(socket, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, optlen));
}

PortType getSocketPort(int fd) {
  PortType listenPort;
  struct ::sockaddr_storage addrStorage;
  socklen_t addrLen = sizeof(addrStorage);
  SYSCHECK_ERR_RETURN_NEG1(getsockname(
      fd, reinterpret_cast<struct ::sockaddr*>(&addrStorage), &addrLen));

  if (addrStorage.ss_family == AF_INET) {
    struct ::sockaddr_in* addr =
        reinterpret_cast<struct ::sockaddr_in*>(&addrStorage);
    listenPort = ntohs(addr->sin_port);

  } else if (addrStorage.ss_family == AF_INET6) { // AF_INET6
    struct ::sockaddr_in6* addr =
        reinterpret_cast<struct ::sockaddr_in6*>(&addrStorage);
    listenPort = ntohs(addr->sin6_port);

  } else {
    throw std::runtime_error("unsupported protocol");
  }
  return listenPort;
}

} // namespace

std::string sockaddrToString(struct ::sockaddr* addr) {
  char address[INET6_ADDRSTRLEN + 1];
  if (addr->sa_family == AF_INET) {
    struct ::sockaddr_in* s = reinterpret_cast<struct ::sockaddr_in*>(addr);
    SYSCHECK(::inet_ntop(AF_INET, &(s->sin_addr), address, INET_ADDRSTRLEN), __output != nullptr)
    address[INET_ADDRSTRLEN] = '\0';
  } else if (addr->sa_family == AF_INET6) {
    struct ::sockaddr_in6* s = reinterpret_cast<struct ::sockaddr_in6*>(addr);
    SYSCHECK(::inet_ntop(AF_INET6, &(s->sin6_addr), address, INET6_ADDRSTRLEN), __output != nullptr)
    address[INET6_ADDRSTRLEN] = '\0';
  } else {
    throw std::runtime_error("unsupported protocol");
  }
  return address;
}

// listen, connect and accept
std::pair<int, PortType> listen(PortType port) {
  struct ::addrinfo hints, *res = NULL;
  std::memset(&hints, 0x00, sizeof(hints));
  hints.ai_flags = AI_PASSIVE | AI_ADDRCONFIG;
  hints.ai_family = AF_UNSPEC; // either IPv4 or IPv6
  hints.ai_socktype = SOCK_STREAM; // TCP

  // `getaddrinfo` will sort addresses according to RFC 3484 and can be tweeked
  //  by editing `/etc/gai.conf`. so there is no need to manual sorting
  // or protocol preference.
  int err = ::getaddrinfo(nullptr, std::to_string(port).data(), &hints, &res);
  if (err != 0 || !res) {
    throw std::invalid_argument(
        "cannot find host to listen on: " + std::string(gai_strerror(err)));
  }

  std::shared_ptr<struct ::addrinfo> addresses(
      res, [](struct ::addrinfo* p) { ::freeaddrinfo(p); });

  struct ::addrinfo* nextAddr = addresses.get();
  int socket;
  while (true) {
    try {
      SYSCHECK_ERR_RETURN_NEG1(
          socket = ::socket(
              nextAddr->ai_family,
              nextAddr->ai_socktype,
              nextAddr->ai_protocol))

      int optval = 1;
      SYSCHECK_ERR_RETURN_NEG1(
          ::setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(int)))

      SYSCHECK_ERR_RETURN_NEG1(::bind(socket, nextAddr->ai_addr, nextAddr->ai_addrlen))
      SYSCHECK_ERR_RETURN_NEG1(::listen(socket, LISTEN_QUEUE_SIZE))
      break;

    } catch (const std::system_error& e) {
      ::close(socket);
      nextAddr = nextAddr->ai_next;

      // we have tried all addresses but could not start
      // listening on any of them
      if (!nextAddr) {
        throw;
      }
    }
  }

  // get listen port and address
  return {socket, getSocketPort(socket)};
}

int connect(
    const std::string& address,
    PortType port,
    bool wait,
    const std::chrono::milliseconds& timeout) {
  struct ::addrinfo hints, *res = NULL;
  std::memset(&hints, 0x00, sizeof(hints));
  hints.ai_flags = AI_NUMERICSERV; // specifies that port (service) is numeric
  hints.ai_family = AF_UNSPEC; // either IPv4 or IPv6
  hints.ai_socktype = SOCK_STREAM; // TCP

  // `getaddrinfo` will sort addresses according to RFC 3484 and can be tweeked
  // by editing `/etc/gai.conf`. so there is no need to manual sorting
  // or protcol preference.
  int err =
      ::getaddrinfo(address.data(), std::to_string(port).data(), &hints, &res);
  if (err != 0 || !res) {
    throw std::invalid_argument(
        "host not found: " + std::string(gai_strerror(err)));
  }

  std::shared_ptr<struct ::addrinfo> addresses(
      res, [](struct ::addrinfo* p) { ::freeaddrinfo(p); });

  struct ::addrinfo* nextAddr = addresses.get();
  int socket;
  // we'll loop over the addresses only if at least of them gave us ECONNREFUSED
  // Maybe the host was up, but the server wasn't running.
  bool anyRefused = false;
  while (true) {
    try {
      SYSCHECK_ERR_RETURN_NEG1(
          socket = ::socket(
              nextAddr->ai_family,
              nextAddr->ai_socktype,
              nextAddr->ai_protocol))

      ResourceGuard socketGuard([socket]() { ::close(socket); });

      // We need to connect in non-blocking mode, so we can use a timeout
      SYSCHECK_ERR_RETURN_NEG1(::fcntl(socket, F_SETFL, O_NONBLOCK));

      int ret = ::connect(socket, nextAddr->ai_addr, nextAddr->ai_addrlen);

      if (ret != 0 && errno != EINPROGRESS) {
        throw std::system_error(errno, std::system_category());
      }

      struct ::pollfd pfd;
      pfd.fd = socket;
      pfd.events = POLLOUT;

      int numReady = ::poll(&pfd, 1, timeout.count());
      if (numReady < 0) {
        throw std::system_error(errno, std::system_category());
      } else if (numReady == 0) {
        errno = 0;
        throw std::runtime_error("connect() timed out");
      }

      socklen_t errLen = sizeof(errno);
      errno = 0;
      ::getsockopt(socket, SOL_SOCKET, SO_ERROR, &errno, &errLen);

      // `errno` is set when:
      //  1. `getsockopt` has failed
      //  2. there is awaiting error in the socket
      //  (the error is saved to the `errno` variable)
      if (errno != 0) {
        throw std::system_error(errno, std::system_category());
      }

      // Disable non-blocking mode
      int flags;
      SYSCHECK_ERR_RETURN_NEG1(flags = ::fcntl(socket, F_GETFL));
      SYSCHECK_ERR_RETURN_NEG1(::fcntl(socket, F_SETFL, flags & (~O_NONBLOCK)));
      socketGuard.release();
      break;

    } catch (std::exception& e) {
      if (errno == ECONNREFUSED) {
        anyRefused = true;
      }

      // We need to move to the next address because this was not available
      // to connect or to create a socket.
      nextAddr = nextAddr->ai_next;

      // We have tried all addresses but could not connect to any of them.
      if (!nextAddr) {
        if (!wait || !anyRefused) {
          throw;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        anyRefused = false;
        nextAddr = addresses.get();
      }
    }
  }

  setSocketNoDelay(socket);

  return socket;
}

std::tuple<int, std::string> accept(
    int listenSocket,
    const std::chrono::milliseconds& timeout) {
  // poll on listen socket, it allows to make timeout
  std::unique_ptr<struct ::pollfd[]> events(new struct ::pollfd[1]);
  events[0] = {.fd = listenSocket, .events = POLLIN};

  while (true) {
    int res = ::poll(events.get(), 1, timeout.count());
    if (res == 0) {
      throw std::runtime_error(
          "waiting for processes to "
          "connect has timed out");
    } else if (res == -1) {
      if (errno == EINTR) {
        continue;
      }
      throw std::system_error(errno, std::system_category());
    } else {
      if (!(events[0].revents & POLLIN))
        throw std::system_error(ECONNABORTED, std::system_category());
      break;
    }
  }

  int socket;
  SYSCHECK_ERR_RETURN_NEG1(socket = ::accept(listenSocket, NULL, NULL))

  // Get address of the connecting process
  struct ::sockaddr_storage addr;
  socklen_t addrLen = sizeof(addr);
  SYSCHECK_ERR_RETURN_NEG1(::getpeername(
      socket, reinterpret_cast<struct ::sockaddr*>(&addr), &addrLen))

  setSocketNoDelay(socket);

  return std::make_tuple(
      socket, sockaddrToString(reinterpret_cast<struct ::sockaddr*>(&addr)));
}

} // namespace tcputil
} // namespace c10d
