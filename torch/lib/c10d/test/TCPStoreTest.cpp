#include <c10d/test/StoreTestCommon.hpp>

#include <cstdlib>
#include <iostream>
#include <thread>

#include <c10d/PrefixStore.hpp>
#include <c10d/TCPStore.hpp>

void testHelper(const std::string& prefix = "") {
  const auto numThreads = 16;
  const auto numWorkers = numThreads + 1;
  // server store
  c10d::TCPStore serverTCPStore("127.0.0.1", 29500, numWorkers, true);
  c10d::PrefixStore serverStore(prefix, serverTCPStore);

  // Basic set/get on the server store
  c10d::test::set(serverStore, "key0", "value0");
  c10d::test::set(serverStore, "key1", "value1");
  c10d::test::set(serverStore, "key2", "value2");
  c10d::test::check(serverStore, "key0", "value0");
  c10d::test::check(serverStore, "key1", "value1");
  c10d::test::check(serverStore, "key2", "value2");

  // Hammer on TCPStore
  std::vector<std::thread> threads;
  const auto numIterations = 1000;
  c10d::test::Semaphore sem1, sem2;

  // Each thread will have a client store to send/recv data
  std::vector<std::unique_ptr<c10d::TCPStore>> clientTCPStores;
  std::vector<std::unique_ptr<c10d::PrefixStore>> clientStores;
  for (auto i = 0; i < numThreads; i++) {
    clientTCPStores.push_back(std::unique_ptr<c10d::TCPStore>(
        new c10d::TCPStore("127.0.0.1", 29500, numWorkers, false)));
    clientStores.push_back(std::unique_ptr<c10d::PrefixStore>(
        new c10d::PrefixStore(prefix, *clientTCPStores[i])));
  }

  std::string expectedCounterRes = std::to_string(numThreads * numIterations);

  for (auto i = 0; i < numThreads; i++) {
    threads.push_back(
        std::thread([&sem1, &sem2, &clientStores, i, &expectedCounterRes] {
          for (auto j = 0; j < numIterations; j++) {
            clientStores[i]->add("counter", 1);
          }
          // Let each thread set and get key on its client store
          std::string key = "thread_" + std::to_string(i);
          for (auto j = 0; j < numIterations; j++) {
            std::string val = "thread_val_" + std::to_string(j);
            c10d::test::set(*clientStores[i], key, val);
            c10d::test::check(*clientStores[i], key, val);
          }

          sem1.post();
          sem2.wait();
          // Check the counter results
          c10d::test::check(*clientStores[i], "counter", expectedCounterRes);
          // Now check other threads' written data
          for (auto j = 0; j < numThreads; j++) {
            if (j == i) {
              continue;
            }
            std::string key = "thread_" + std::to_string(i);
            std::string val = "thread_val_" + std::to_string(numIterations - 1);
            c10d::test::check(*clientStores[i], key, val);
          }
        }));
  }

  sem1.wait(numThreads);
  sem2.post(numThreads);

  for (auto& thread : threads) {
    thread.join();
  }

  // Clear the store to test that client disconnect won't shutdown the store
  clientStores.clear();
  clientTCPStores.clear();

  // Check that the counter has the expected value
  c10d::test::check(serverStore, "counter", expectedCounterRes);

  // Check that each threads' written data from the main thread
  for (auto i = 0; i < numThreads; i++) {
    std::string key = "thread_" + std::to_string(i);
    std::string val = "thread_val_" + std::to_string(numIterations - 1);
    c10d::test::check(serverStore, key, val);
  }
}
int main(int argc, char** argv) {
  testHelper();
  testHelper("testPrefix");
  std::cout << "Test succeeded" << std::endl;
  return EXIT_SUCCESS;
}
