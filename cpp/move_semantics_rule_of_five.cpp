// move_semantics_rule_of_five.cpp
// Companion example for: topnotchnote.com/cpp/move_semantics.html
//
// All five special member functions on one class, each printing a message,
// so you can see exactly which one runs for each line in main().
// Compile:  g++ -Wall -Wextra -std=c++17 -o main move_semantics_rule_of_five.cpp

#include <iostream>
#include <string>
#include <utility>

class Animal {
  std::string name, sound;

public:
  Animal() : name("unknown"), sound("unknown") {
    std::cout << "  default ctor\n";
  }

  Animal(std::string name, std::string sound) : name{std::move(name)}, sound{std::move(sound)} {
    std::cout << "  parameterized ctor (" << this->name << ")\n";
  }

  // rule of three -----------------------------------------------------
  Animal(const Animal &rhs) : name(rhs.name), sound(rhs.sound) {
    std::cout << "  copy ctor (" << name << ")\n";
  }

  Animal &operator=(const Animal &rhs) {
    std::cout << "  copy assignment (" << rhs.name << ")\n";
    if (this != &rhs) {
      name = rhs.name;
      sound = rhs.sound;
    }
    return *this;
  }

  ~Animal() { std::cout << "  dtor (" << name << ")\n"; }

  // rule of five adds these two ----------------------------------------
  Animal(Animal &&rhs) noexcept : name(std::move(rhs.name)), sound(std::move(rhs.sound)) {
    std::cout << "  move ctor\n";
    rhs.name = "(moved-from)";
  }

  Animal &operator=(Animal &&rhs) noexcept {
    std::cout << "  move assignment\n";
    if (this != &rhs) {
      name = std::move(rhs.name);
      sound = std::move(rhs.sound);
      rhs.name = "(moved-from)";
    }
    return *this;
  }

  const std::string &getName() const { return name; }
};

int main() {
  std::cout << "Animal a;\n";
  Animal a;

  std::cout << "\nAnimal b(\"dog\", \"bark\");\n";
  Animal b("dog", "bark");

  std::cout << "\nAnimal c(b);            // copy ctor\n";
  Animal c(b);

  std::cout << "\nAnimal d = b;            // ALSO copy ctor -- sugar for Animal d(b)\n";
  Animal d = b;

  std::cout << "\nAnimal e;\ne = b;                  // default ctor, then copy assignment\n";
  Animal e;
  e = b;

  std::cout << "\nAnimal f(std::move(b));  // move ctor -- b is left valid-but-empty\n";
  Animal f(std::move(b));
  std::cout << "  b's name is now: " << b.getName() << "\n";

  std::cout << "\n-- leaving main, destructors run for a..f --\n";
  return 0;
}
