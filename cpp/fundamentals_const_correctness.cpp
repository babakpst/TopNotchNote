// fundamentals_const_correctness.cpp
// Companion example for: topnotchnote.com/cpp/fundamentals.html
//
// const member functions, const objects, and the mutable escape hatch.
// Compile:  g++ -Wall -Wextra -std=c++17 -o main fundamentals_const_correctness.cpp

#include <iostream>

class Account {
  mutable int accessCount = 0;   // allowed to change even from a const function
  double balance = 100.0;

public:
  double getBalance() const {
    accessCount++;                // OK -- accessCount is mutable
    // balance = 0;                 // would NOT compile: assignment of member in read-only object
    return balance;
  }

  int timesRead() const { return accessCount; }

  void deposit(double amount) {   // non-const: allowed to modify balance
    balance += amount;
  }
};

int main() {
  Account a;
  std::cout << "balance: " << a.getBalance() << "\n";
  std::cout << "balance: " << a.getBalance() << "\n";
  std::cout << "accessCount (mutated through a const function): " << a.timesRead() << "\n";

  const Account frozen;          // a const object
  std::cout << "frozen balance: " << frozen.getBalance() << "\n";  // OK -- getBalance() is const
  // frozen.deposit(10);            // would NOT compile: deposit() isn't const, frozen is

  return 0;
}
