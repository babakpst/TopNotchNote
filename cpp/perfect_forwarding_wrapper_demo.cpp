// perfect_forwarding_wrapper_demo.cpp
// Companion example for: topnotchnote.com/cpp/perfect_forwarding.html
//
// A wrapper function template that forwards its argument on to a
// target function -- once with a named variable (lvalue), once with
// a temporary (rvalue) -- so you can see the copy constructor and
// move constructor each fire on exactly the call that should trigger
// them, and reference collapsing confirmed with static_assert.
// Compile:  g++ -Wall -Wextra -std=c++17 -o main perfect_forwarding_wrapper_demo.cpp

#include <iostream>
#include <type_traits>
#include <utility>

class Widget {
  std::string tag;
public:
  explicit Widget(std::string tag) : tag(std::move(tag)) {
    std::cout << "  Widget(\"" << this->tag << "\") ctor\n";
  }
  Widget(const Widget &rhs) : tag(rhs.tag) {
    std::cout << "  Widget copy ctor (" << tag << ")\n";
  }
  Widget(Widget &&rhs) noexcept : tag(std::move(rhs.tag)) {
    std::cout << "  Widget move ctor (" << tag << ")\n";
  }
};

void target(const Widget &w) { (void)w; std::cout << "  target(const Widget&) -- bound without copying\n"; }
void target(Widget &&w) { Widget taken(std::move(w)); std::cout << "  target(Widget&&) -- moved into a local\n"; }

template <typename T>
void wrapper(T &&arg) {
  // T is deduced here: Widget& for an lvalue argument, Widget for an rvalue argument.
  // Reference collapsing turns the parameter's declared T&& into Widget& or Widget&&, respectively.
  static_assert(std::is_lvalue_reference<T>::value || std::is_rvalue_reference<T&&>::value, "sanity check");
  target(std::forward<T>(arg));
}

int main() {
  std::cout << "-- forwarding an lvalue --\n";
  Widget named("named");
  wrapper(named);   // T deduced as Widget& -> forwarded as an lvalue -> target(const Widget&)

  std::cout << "\n-- forwarding an rvalue --\n";
  wrapper(Widget("temporary"));   // T deduced as Widget -> forwarded as an rvalue -> target(Widget&&)

  std::cout << "\n-- reference collapsing, confirmed at compile time --\n";
  static_assert(std::is_same<typename std::add_rvalue_reference<Widget &>::type, Widget &>::value,
                "Widget& && collapses to Widget&");
  static_assert(std::is_same<typename std::add_rvalue_reference<Widget &&>::type, Widget &&>::value,
                "Widget&& && collapses to Widget&&");
  std::cout << "  both static_asserts above passed at compile time\n";

  return 0;
}
