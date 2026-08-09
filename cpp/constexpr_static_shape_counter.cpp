// constexpr_static_shape_counter.cpp
// Companion example for: topnotchnote.com/cpp/constexpr_static.html
//
// A base class tracks how many objects of it -- and every subclass --
// have ever been constructed, using one static counter shared across
// the whole hierarchy. Run it to see the same shared count reported
// from three different objects, including a Circle that never
// declares a counter of its own.
// Compile:  g++ -Wall -Wextra -std=c++17 -o main constexpr_static_shape_counter.cpp

#include <iostream>
#include <string>

class Shape {
public:
  Shape() { total++; }
  static int total;   // declaration -- one copy, shared by every Shape and every subclass

  void report(const std::string &label) const {
    std::cout << "  " << label << ".total = " << total << "\n";
  }
};
int Shape::total = 0;   // definition -- required exactly once, outside the class

class Circle : public Shape {};   // doesn't declare its own total -- shares Shape's

int main() {
  std::cout << "Shape a, b;\nCircle c;\n\n";
  Shape a, b;
  Circle c;   // Circle's implicit ctor calls Shape's, incrementing the shared counter too

  a.report("a");
  b.report("b");
  c.report("c");   // same shared counter, not one per subclass

  std::cout << "\nShape::total accessed directly: " << Shape::total << "\n";
  return 0;
}
