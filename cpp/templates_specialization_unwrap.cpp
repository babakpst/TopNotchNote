// templates_specialization_magnitude.cpp
// Companion example for: topnotchnote.com/cpp/templates_generics.html
//
// A trait that answers "what type is this actually wrapping?" -- the
// generic case says a type wraps itself; a partial specialization for
// std::optional<T> says it wraps a plain T. Same pattern as picking the
// right comparison/formatting logic per type category, resolved entirely
// at compile time.
// Compile:  g++ -Wall -Wextra -std=c++17 -o main templates_specialization_magnitude.cpp

#include <iostream>
#include <optional>
#include <string>

// Generic case: an ordinary type wraps itself -- there's nothing to unwrap.
template <typename T>
struct Unwrap {
  using type = T;
};

// Partial specialization: std::optional<T> wraps a T underneath.
template <typename T>
struct Unwrap<std::optional<T>> {
  using type = T;
};

template <typename T>
void describe(const T &value) {
  using Inner = typename Unwrap<T>::type;
  std::cout << "sizeof(T) = " << sizeof(value) << " bytes, "
            << "sizeof(unwrapped type) = " << sizeof(Inner) << " bytes\n";
}

int main() {
  int plainNumber = 42;
  std::cout << "-- plain int --\n";
  describe(plainNumber);                 // Unwrap<int>::type is int

  std::optional<double> maybePrice = 19.99;
  std::cout << "\n-- std::optional<double> --\n";
  describe(maybePrice);                  // Unwrap<optional<double>>::type is double, not optional<double>

  std::optional<char> empty;
  std::cout << "\n-- an EMPTY std::optional<char> --\n";
  std::cout << "has_value: " << empty.has_value() << "\n";
  describe(empty);                       // the specialization still resolves at compile time, regardless of runtime emptiness

  return 0;
}
