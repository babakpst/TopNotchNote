// operator_overloading_binary_forms.cpp
// Companion example for: topnotchnote.com/cpp/operator_overloading.html
//
// Four different ways to implement operator+ for the same class -- a free
// function, a member function, a friend function, and a friend declared
// inside the class but defined outside it. All four produce identical
// behavior for `e1 + e2`; only where the code lives and what it can
// access differs. Each variant lives in its own namespace so all four
// can coexist in one file.
// Compile:  g++ -Wall -Wextra -std=c++17 -o main operator_overloading_binary_forms.cpp

#include <iostream>

// ---------------------------------------------------------------------
// 1. Free (non-member, non-friend) function -- only touches public members
// ---------------------------------------------------------------------
namespace as_free_function {
class Complex {
public:
  double real, imag;   // public here so the free function below can reach them
  Complex(double r = 0, double i = 0) : real(r), imag(i) {}
};

Complex operator+(const Complex &lhs, const Complex &rhs) {
  return Complex(lhs.real + rhs.real, lhs.imag + rhs.imag);
}
}

// ---------------------------------------------------------------------
// 2. Member function -- left operand is implicitly *this
// ---------------------------------------------------------------------
namespace as_member_function {
class Complex {
  double real, imag;
public:
  Complex(double r = 0, double i = 0) : real(r), imag(i) {}
  Complex operator+(const Complex &rhs) const {
    return Complex(real + rhs.real, imag + rhs.imag);
  }
  double getReal() const { return real; }
  double getImag() const { return imag; }
};
}

// ---------------------------------------------------------------------
// 3. Friend function, defined inline inside the class body
// ---------------------------------------------------------------------
namespace as_friend_inline {
class Complex {
  double real, imag;
public:
  Complex(double r = 0, double i = 0) : real(r), imag(i) {}
  friend Complex operator+(const Complex &lhs, const Complex &rhs) {
    return Complex(lhs.real + rhs.real, lhs.imag + rhs.imag);   // friend: can reach private members
  }
  double getReal() const { return real; }
  double getImag() const { return imag; }
};
}

// ---------------------------------------------------------------------
// 4. Friend declared in the class, defined outside it -- same behavior as #3
// ---------------------------------------------------------------------
namespace as_friend_outside {
class Complex {
  double real, imag;
public:
  Complex(double r = 0, double i = 0) : real(r), imag(i) {}
  friend Complex operator+(const Complex &lhs, const Complex &rhs);   // declaration only
  double getReal() const { return real; }
  double getImag() const { return imag; }
};

Complex operator+(const Complex &lhs, const Complex &rhs) {
  return Complex(lhs.real + rhs.real, lhs.imag + rhs.imag);
}
}

int main() {
  {
    using namespace as_free_function;
    Complex e1(1, 2), e2(3, 4);
    Complex sum = e1 + e2;
    std::cout << "free function:      " << sum.real << " + " << sum.imag << "i\n";
  }
  {
    using namespace as_member_function;
    Complex e1(1, 2), e2(3, 4);
    Complex sum = e1 + e2;
    std::cout << "member function:    " << sum.getReal() << " + " << sum.getImag() << "i\n";
  }
  {
    using namespace as_friend_inline;
    Complex e1(1, 2), e2(3, 4);
    Complex sum = e1 + e2;
    std::cout << "friend (inline):    " << sum.getReal() << " + " << sum.getImag() << "i\n";
  }
  {
    using namespace as_friend_outside;
    Complex e1(1, 2), e2(3, 4);
    Complex sum = e1 + e2;
    std::cout << "friend (outside):   " << sum.getReal() << " + " << sum.getImag() << "i\n";
  }
  return 0;
}
