// oop_relationships_composition_aggregation.cpp
// Companion example for: topnotchnote.com/cpp/oop_relationships.html
//
// Composition (Car owns its Engine outright) side by side with
// aggregation (Department borrows a Teacher it doesn't own) -- traced
// with constructor/destructor output so the lifetime difference is
// visible, not just asserted.
// Compile:  g++ -Wall -Wextra -std=c++17 -o main oop_relationships_composition_aggregation.cpp

#include <iostream>
#include <string>

class Engine {
public:
  explicit Engine(int id) { std::cout << "    Engine(" << id << ") ctor\n"; }
  ~Engine() { std::cout << "    Engine dtor\n"; }
};

class Car {   // COMPOSITION -- Car owns its Engine outright
  Engine engine;   // held by value: constructed with Car, destroyed with Car
public:
  explicit Car(int id) : engine(id) {}
};

class Teacher {
public:
  explicit Teacher(std::string n) : name(std::move(n)) {
    std::cout << "  Teacher(" << name << ") ctor\n";
  }
  ~Teacher() { std::cout << "  Teacher(" << name << ") dtor\n"; }
  std::string name;
};

class Department {   // AGGREGATION -- Department borrows a Teacher it doesn't own
  Teacher *teacher;   // just a pointer to something created elsewhere
public:
  explicit Department(Teacher *t) : teacher(t) {
    std::cout << "  Department borrowing " << teacher->name << "\n";
  }
  ~Department() { std::cout << "  Department dtor (deliberately does NOT delete teacher)\n"; }
};

int main() {
  std::cout << "-- composition --\n";
  {
    std::cout << "  Car c(1);\n";
    Car c(1);   // prints Engine ctor
  }               // c goes out of scope -- prints Engine dtor automatically
  std::cout << "  (Car and its Engine are both gone here)\n";

  std::cout << "\n-- aggregation --\n";
  Teacher *t = new Teacher("Dr. Lee");   // created outside Department's control
  {
    Department d(t);   // d borrows t
  }                      // d is destroyed -- t is untouched, still valid
  std::cout << "  Department is gone, but Dr. Lee is still alive: " << t->name << "\n";
  delete t;   // whoever created t is responsible for destroying it

  return 0;
}
