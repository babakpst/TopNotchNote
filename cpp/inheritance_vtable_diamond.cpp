// inheritance_vtable_diamond.cpp
// Companion example for: topnotchnote.com/cpp/inheritance_polymorphism.html
//
// Part 1: confirms the vptr's 8-byte cost via sizeof.
// Part 2: the diamond problem, and how virtual inheritance fixes it.
// Compile:  g++ -Wall -Wextra -std=c++17 -o main inheritance_vtable_diamond.cpp

#include <iostream>

// ---------------------------------------------------------------------
// Part 1: sizeof cost of a vtable pointer
// ---------------------------------------------------------------------
class NoVirtual {
  int *ptr;
  float f;
  bool b;
public:
  void print() {}
};

class WithVirtual {
  int *ptr;
  float f;
  bool b;
public:
  virtual void print() { std::cout << "print\n"; }
};

// ---------------------------------------------------------------------
// Part 2: the diamond problem
// ---------------------------------------------------------------------
namespace without_virtual_inheritance {
class PoweredDevice {
public:
  PoweredDevice() { std::cout << "  PoweredDevice ctor\n"; }
};
class Scanner : public PoweredDevice {};
class Printer : public PoweredDevice {};
class AllInOnePrinter : public Scanner, public Printer {};   // has TWO PoweredDevice subobjects
}

namespace with_virtual_inheritance {
class PoweredDevice {
public:
  PoweredDevice() { std::cout << "  PoweredDevice ctor\n"; }
};
class Scanner : virtual public PoweredDevice {};
class Printer : virtual public PoweredDevice {};
class AllInOnePrinter : public Scanner, public Printer {};   // exactly ONE shared PoweredDevice subobject
}

int main() {
  std::cout << "-- Part 1: vptr cost --\n";
  std::cout << "sizeof(NoVirtual)   = " << sizeof(NoVirtual) << " bytes\n";
  std::cout << "sizeof(WithVirtual) = " << sizeof(WithVirtual)
            << " bytes (+8 for the vptr, same data members otherwise)\n";

  std::cout << "\n-- Part 2a: diamond WITHOUT virtual inheritance --\n";
  std::cout << "constructing an AllInOnePrinter:\n";
  {
    using namespace without_virtual_inheritance;
    AllInOnePrinter printer;   // prints "PoweredDevice ctor" TWICE -- one per inheritance path
  }

  std::cout << "\n-- Part 2b: diamond WITH virtual inheritance --\n";
  std::cout << "constructing an AllInOnePrinter:\n";
  {
    using namespace with_virtual_inheritance;
    AllInOnePrinter printer;   // prints "PoweredDevice ctor" ONCE -- a single shared subobject
  }

  return 0;
}
