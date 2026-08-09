// testing_bug_patterns_demo.cpp
// Companion example for: topnotchnote.com/cpp/testing_and_tooling.html
//
// Three bugs a static analyzer catches that the compiler often
// doesn't: use-after-free, an uninitialized read, and an out-of-bounds
// write. The buggy functions are deliberately never called from
// main() -- this file exists to be handed to cppcheck, not executed:
//   cppcheck --enable=all testing_bug_patterns_demo.cpp
// Compile (builds cleanly despite the bugs -- that's the point):
//   g++ -Wall -Wextra -std=c++17 -c testing_bug_patterns_demo.cpp

#include <iostream>

void useAfterFree(int *p) {
  delete p;
  int j = *p;   // (1) use-after-free -- p was just deleted
  std::cout << j;
}

void uninitializedRead() {
  int uninitialized;
  if (uninitialized == uninitialized) {   // (2) reading a variable that was never given a value
    std::cout << "always true, for the wrong reason\n";
  }
}

void outOfBounds() {
  int ages[3];
  ages[0] = 18; ages[1] = 21; ages[2] = 35;
  ages[3] = 40;   // (3) out-of-bounds write -- valid indices are 0..2
}

int main() {
  std::cout << "This file is meant to be analyzed, not run -- see the header comment.\n";
  std::cout << "The three buggy functions above are intentionally never called here.\n";
  return 0;
}
