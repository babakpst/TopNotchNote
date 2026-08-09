// design_patterns_singleton_logger.cpp
// Companion example for: topnotchnote.com/cpp/design_patterns.html
//
// The modern Meyers-singleton idiom: a function-local static, whose
// construction is guaranteed thread-safe since C++11 -- no manual
// locking, no raw new'd pointer, no leak at exit.
// Compile:  g++ -Wall -Wextra -std=c++17 -o main design_patterns_singleton_logger.cpp

#include <iostream>
#include <string>

class Logger {
  Logger() { std::cout << "  Logger constructed (first instance() call)\n"; }

public:
  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

  static Logger &instance() {
    static Logger single;   // constructed on first call, guaranteed thread-safe since C++11
    return single;           // destructed automatically at program exit -- no manual delete
  }

  void log(const std::string &msg) { std::cout << "  [log] " << msg << "\n"; }
};

int main() {
  std::cout << "main() starts -- Logger has not been constructed yet\n\n";

  std::cout << "Logger::instance().log(\"started\");\n";
  Logger::instance().log("started");   // first call -- constructs the single instance here

  std::cout << "\nLogger::instance().log(\"still the same instance\");\n";
  Logger::instance().log("still the same instance");   // no second construction

  std::cout << "\n&Logger::instance() is the same address both times: "
            << (&Logger::instance() == &Logger::instance() ? "true" : "false") << "\n";
  return 0;
}
