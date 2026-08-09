// classes_constructors_bankaccount_rule_of_three.cpp
// Companion example for: topnotchnote.com/cpp/classes_constructors.html
//
// A class with a constructor-parameter-shadowing bug fixed via this->,
// plus the full rule of three (copy ctor, copy assignment with a
// self-assignment guard, destructor) traced with output so you can see
// exactly which member function runs for each line in main().
// Compile:  g++ -Wall -Wextra -std=c++17 -o main classes_constructors_bankaccount_rule_of_three.cpp

#include <iostream>
#include <string>

class BankAccount {
  std::string owner;
  double balance;

public:
  BankAccount(std::string owner, double balance) : owner(owner) {
    // BUG avoided here: "balance = balance;" would assign the parameter
    // to itself and leave the member uninitialized -- this-> disambiguates
    this->balance = balance;
    std::cout << "  ctor: " << this->owner << " opens with $" << this->balance << "\n";
  }

  BankAccount(const BankAccount &rhs) : owner(rhs.owner), balance(rhs.balance) {
    std::cout << "  copy ctor: cloned " << owner << "'s account\n";
  }

  BankAccount &operator=(const BankAccount &rhs) {
    std::cout << "  copy assignment: " << owner << " := " << rhs.owner << "\n";
    if (this != &rhs) {   // guards against a = a corrupting balance via self-copy
      owner = rhs.owner;
      balance = rhs.balance;
    }
    return *this;
  }

  ~BankAccount() { std::cout << "  dtor: closing " << owner << "'s account\n"; }

  void deposit(double amount) { balance += amount; }
  double getBalance() const { return balance; }
  const std::string &getOwner() const { return owner; }
};

int main() {
  std::cout << "BankAccount a(\"Alice\", 100.0);\n";
  BankAccount a("Alice", 100.0);
  a.deposit(50.0);

  std::cout << "\nBankAccount b(a);           // copy ctor\n";
  BankAccount b(a);

  std::cout << "\nBankAccount c(\"Carol\", 0.0);\nc = a;                       // copy assignment\n";
  BankAccount c("Carol", 0.0);
  c = a;

  std::cout << "\nb's balance after copying a: $" << b.getBalance() << "\n";
  std::cout << "c's owner after assignment: " << c.getOwner() << "\n";

  std::cout << "\n-- leaving main, destructors run for c, b, a --\n";
  return 0;
}
