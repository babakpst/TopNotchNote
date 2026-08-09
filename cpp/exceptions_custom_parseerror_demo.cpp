// exceptions_custom_parseerror_demo.cpp
// Companion example for: topnotchnote.com/cpp/exceptions_file_io.html
//
// A custom exception type derived from std::runtime_error, thrown from
// deep inside a call chain and caught two different ways -- once
// specifically, once generically -- to show catch-clause ordering in
// action along with stack unwinding running local destructors on the
// way out.
// Compile:  g++ -Wall -Wextra -std=c++17 -o main exceptions_custom_parseerror_demo.cpp

#include <iostream>
#include <stdexcept>
#include <string>

class ParseError : public std::runtime_error {
public:
  explicit ParseError(const std::string &msg) : std::runtime_error(msg) {}
};

// stands in for any local object whose destructor must run during unwinding
class ScopeTracer {
  std::string name;
public:
  explicit ScopeTracer(std::string name) : name(std::move(name)) {
    std::cout << "  entering " << this->name << "\n";
  }
  ~ScopeTracer() { std::cout << "  leaving " << name << " (destructor runs during unwind)\n"; }
};

int parseToken(const std::string &token) {
  ScopeTracer trace("parseToken");
  if (token.empty()) throw ParseError("unexpected empty token");
  return token.size();
}

int parseLine(const std::string &line) {
  ScopeTracer trace("parseLine");
  return parseToken(line);   // ParseError thrown here unwinds through this frame too
}

int main() {
  std::cout << "-- case 1: caught specifically as ParseError --\n";
  try {
    parseLine("");
  } catch (const ParseError &e) {
    std::cout << "caught ParseError: " << e.what() << "\n";
  } catch (const std::exception &e) {
    std::cout << "caught some other std::exception: " << e.what() << "\n";
  }

  std::cout << "\n-- case 2: same exception, but only a generic handler is present --\n";
  try {
    parseLine("");
  } catch (const std::exception &e) {
    // ParseError IS-A runtime_error IS-A exception -- this still catches it
    std::cout << "caught via std::exception&: " << e.what() << "\n";
  }

  return 0;
}
