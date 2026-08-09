// concurrency_counter_race_demo.cpp
// Companion example for: topnotchnote.com/cpp/concurrency.html
//
// Four threads increment a plain int a few hundred thousand times
// each with no synchronization, then four threads do the same thing
// through a std::mutex. The unprotected total reliably falls short of
// the expected count; the protected one never does.
// Compile:  g++ -Wall -Wextra -std=c++17 -pthread -o main concurrency_counter_race_demo.cpp

#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

constexpr int kThreads = 4;
constexpr int kIncrementsPerThread = 200000;

void incrementUnsafe(int &counter) {
  for (int i = 0; i < kIncrementsPerThread; ++i) {
    ++counter;   // read-modify-write, not atomic -- a data race when multiple threads do this concurrently
  }
}

void incrementSafe(int &counter, std::mutex &m) {
  for (int i = 0; i < kIncrementsPerThread; ++i) {
    std::lock_guard<std::mutex> guard(m);
    ++counter;
  }
}

int main() {
  const int expected = kThreads * kIncrementsPerThread;

  std::cout << "-- unprotected: " << kThreads << " threads, no synchronization --\n";
  int unsafeCounter = 0;
  {
    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; ++i) threads.emplace_back(incrementUnsafe, std::ref(unsafeCounter));
    for (auto &t : threads) t.join();
  }
  std::cout << "  expected: " << expected << "\n";
  std::cout << "  actual:   " << unsafeCounter
            << (unsafeCounter == expected ? "  (got lucky this run)" : "  (lost increments to the race)") << "\n";

  std::cout << "\n-- protected: same work, guarded by a std::mutex --\n";
  int safeCounter = 0;
  std::mutex m;
  {
    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; ++i) threads.emplace_back(incrementSafe, std::ref(safeCounter), std::ref(m));
    for (auto &t : threads) t.join();
  }
  std::cout << "  expected: " << expected << "\n";
  std::cout << "  actual:   " << safeCounter << "  (always matches -- every increment is serialized)\n";

  return 0;
}
