// casting_four_casts_demo.cpp
// Companion example for: topnotchnote.com/cpp/casting.html
//
// All four C++-style casts in one program, each doing exactly the one
// thing it's meant for -- run it and see which checks happen at
// compile time, which happen at runtime, and which happen not at all.
// Compile:  g++ -Wall -Wextra -std=c++17 -o main casting_four_casts_demo.cpp

#include <iostream>

class Base {
public:
  virtual ~Base() = default;
};
class Derived : public Base {
public:
  void speak() const { std::cout << "  Derived::speak()\n"; }
};
class Unrelated : public Base {};

void legacyPrint(char *s) { std::cout << "  legacyPrint: " << s << "\n"; }

int main() {
  // static_cast -- compile-time only, no runtime check
  std::cout << "static_cast:\n";
  double d = 3.9;
  int i = static_cast<int>(d);
  std::cout << "  static_cast<int>(3.9) = " << i << "\n";

  // dynamic_cast -- runtime-checked downcast through a polymorphic hierarchy
  std::cout << "\ndynamic_cast:\n";
  Base *pbd = new Derived();
  Base *pbu = new Unrelated();

  if (Derived *good = dynamic_cast<Derived *>(pbd)) {
    std::cout << "  pbd really is a Derived:\n";
    good->speak();
  }
  if (Derived *bad = dynamic_cast<Derived *>(pbu)) {
    bad->speak();
  } else {
    std::cout << "  pbu is not a Derived -- dynamic_cast returned nullptr\n";
  }

  // const_cast -- adds/removes const, nothing else
  std::cout << "\nconst_cast:\n";
  const std::string msg = "hello from a const string";
  legacyPrint(const_cast<char *>(msg.c_str()));   // legacyPrint only reads -- safe here

  // reinterpret_cast -- reinterprets the same bits as an unrelated type
  std::cout << "\nreinterpret_cast:\n";
  int magic = 0x41424344;
  unsigned char *bytes = reinterpret_cast<unsigned char *>(&magic);
  std::cout << "  bytes of 0x41424344: ";
  for (std::size_t b = 0; b < sizeof(magic); ++b)
    std::cout << std::hex << static_cast<int>(bytes[b]) << " ";
  std::cout << std::dec << "\n";

  delete pbd;
  delete pbu;
  return 0;
}
