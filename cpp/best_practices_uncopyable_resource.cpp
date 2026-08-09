// best_practices_uncopyable_resource.cpp
// Companion example for: topnotchnote.com/cpp/best_practices.html
//
// A resource-owning class made uncopyable the modern way (= delete),
// tracing acquire/release so you can see exactly when the resource
// lives and dies -- and that a move still works even though copying
// doesn't.
// Compile:  g++ -Wall -Wextra -std=c++17 -o main best_practices_uncopyable_resource.cpp

#include <iostream>
#include <string>
#include <utility>

class Connection {
  std::string endpoint;

public:
  explicit Connection(std::string endpoint) : endpoint(std::move(endpoint)) {
    std::cout << "  opened connection to " << this->endpoint << "\n";
  }

  ~Connection() {
    if (!endpoint.empty()) std::cout << "  closed connection to " << endpoint << "\n";
  }

  // modern uncopyable idiom -- no hand-rolled private base class needed
  Connection(const Connection &) = delete;
  Connection &operator=(const Connection &) = delete;

  // still movable: the old object is left with an empty endpoint so its
  // destructor knows there's nothing left to close
  Connection(Connection &&rhs) noexcept : endpoint(std::move(rhs.endpoint)) {
    std::cout << "  moved connection to " << endpoint << "\n";
    rhs.endpoint.clear();
  }

  Connection &operator=(Connection &&rhs) noexcept {
    if (this != &rhs) {
      endpoint = std::move(rhs.endpoint);
      rhs.endpoint.clear();
    }
    return *this;
  }
};

Connection openPrimary() { return Connection("db-primary:5432"); }

int main() {
  std::cout << "Connection a(\"db-primary:5432\");\n";
  Connection a("db-primary:5432");

  // Connection b = a;                 // would not compile: copy ctor is deleted
  // a = a;                            // would not compile: copy assignment is deleted

  std::cout << "\nConnection b(std::move(a));\n";
  Connection b(std::move(a));   // fine -- ownership transfers, a is left empty

  std::cout << "\nConnection c = openPrimary();  // guaranteed copy elision, no move even needed\n";
  Connection c = openPrimary();

  std::cout << "\n-- leaving main --\n";
  return 0;
}
