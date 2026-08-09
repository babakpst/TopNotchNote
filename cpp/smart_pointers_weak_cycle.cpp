// smart_pointers_weak_cycle.cpp
// Companion example for: topnotchnote.com/cpp/smart_pointers.html
//
// Two shared_ptrs pointing at each other never reach a zero owner count --
// each keeps the other alive forever. Replacing one side with a weak_ptr
// breaks the cycle.
// Compile:  g++ -Wall -Wextra -std=c++17 -o main smart_pointers_weak_cycle.cpp

#include <iostream>
#include <memory>

struct Child;

struct Parent {
  std::string name;
  std::shared_ptr<Child> child;      // Parent owns Child
  ~Parent() { std::cout << name << " destroyed\n"; }
};

struct Child {
  std::string name;
  std::weak_ptr<Parent> parent;      // Child only OBSERVES Parent -- breaks the cycle
  ~Child() { std::cout << name << " destroyed\n"; }
};

// If Child::parent were a std::shared_ptr<Parent> instead of a weak_ptr,
// neither destructor above would ever print -- both objects would leak,
// because each one's reference count would never reach zero.

int main() {
  std::cout << "-- creating Parent and Child, linked both ways --\n";
  auto parent = std::make_shared<Parent>();
  parent->name = "Parent";
  auto child = std::make_shared<Child>();
  child->name = "Child";

  parent->child = child;        // Parent -> Child: a real (owning) shared_ptr
  child->parent = parent;       // Child -> Parent: a weak_ptr, does NOT bump Parent's owner count

  std::cout << "parent use_count: " << parent.use_count() << " (still 1 -- the weak_ptr doesn't count)\n";

  std::cout << "\n-- checking the back-reference through .lock() --\n";
  if (auto p = child->parent.lock()) {
    std::cout << "child->parent resolved to: " << p->name << "\n";
  }

  std::cout << "\n-- leaving main: both Parent and Child are destroyed cleanly --\n";
  return 0;
}
