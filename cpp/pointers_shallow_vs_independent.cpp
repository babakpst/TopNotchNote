// pointers_shallow_vs_independent.cpp
// Companion example for: topnotchnote.com/cpp/pointers_references.html
//
// Two pointers to the SAME memory (shallow copy) vs. two independently
// allocated objects that merely hold equal values.
// Compile:  g++ -Wall -Wextra -std=c++17 -o main pointers_shallow_vs_independent.cpp

#include <iostream>

int main() {
  // --- shallow copy: ptr2 points at the SAME int as ptr ---
  int i = 5;
  int *ptr = &i;
  int *ptr2 = ptr;          // ptr2 is a copy of the ADDRESS, not the value

  std::cout << "-- shallow copy (same address) --\n";
  std::cout << "before: *ptr=" << *ptr << " *ptr2=" << *ptr2 << "\n";
  *ptr = 42;
  std::cout << "after *ptr = 42: *ptr=" << *ptr << " *ptr2=" << *ptr2
            << " (both changed -- same memory)\n\n";

  // --- NOT a shallow copy: two separate allocations with equal values ---
  int j = 10;
  int *independent1 = new int(j);
  int *independent2 = new int(j);   // a different address, same starting value

  std::cout << "-- independent allocations (equal value, different address) --\n";
  std::cout << "before: *independent1=" << *independent1
            << " *independent2=" << *independent2 << "\n";
  *independent1 = 99;
  std::cout << "after *independent1 = 99: *independent1=" << *independent1
            << " *independent2=" << *independent2
            << " (only the first changed -- separate memory)\n";

  delete independent1;
  delete independent2;
  return 0;
}
