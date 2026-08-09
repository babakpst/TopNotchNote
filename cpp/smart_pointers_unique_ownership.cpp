// smart_pointers_unique_ownership.cpp
// Companion example for: topnotchnote.com/cpp/smart_pointers.html
//
// Two "task slots" passing ownership of a Task back and forth, showing
// make_unique, std::move, get() (borrow) vs. release() (give up ownership),
// and the automatic destruction that happens on reassignment.
// Compile:  g++ -Wall -Wextra -std=c++17 -o main smart_pointers_unique_ownership.cpp

#include <iostream>
#include <memory>
#include <string>

struct Task {
  std::string label;
  explicit Task(std::string label) : label(std::move(label)) {
    std::cout << "  [Task \"" << this->label << "\" created]\n";
  }
  ~Task() { std::cout << "  [Task \"" << label << "\" destroyed]\n"; }
};

void peek(const std::unique_ptr<Task> &slot, const char *name) {
  if (slot) std::cout << "  " << name << " holds: " << slot->label << "\n";
  else      std::cout << "  " << name << " is empty\n";
}

int main() {
  std::cout << "-- two empty task slots --\n";
  std::unique_ptr<Task> primary;
  std::unique_ptr<Task> backup;
  peek(primary, "primary"); peek(backup, "backup");

  std::cout << "\n-- primary takes ownership of a new Task --\n";
  primary = std::make_unique<Task>("render frame");
  peek(primary, "primary"); peek(backup, "backup");

  std::cout << "\n-- backup = std::move(primary): ownership transfers, primary is now empty --\n";
  backup = std::move(primary);
  peek(primary, "primary"); peek(backup, "backup");

  std::cout << "\n-- Task *raw = backup.get(): borrow the address without taking ownership --\n";
  Task *raw = backup.get();
  std::cout << "  raw->label = " << raw->label << " (backup still owns it)\n";

  std::cout << "\n-- primary = make_unique<Task>(\"encode audio\"): fills the empty slot --\n";
  primary = std::make_unique<Task>("encode audio");
  peek(primary, "primary");

  std::cout << "\n-- primary = make_unique<Task>(\"upload result\"): the OLD Task is destroyed automatically first --\n";
  primary = std::make_unique<Task>("upload result");
  peek(primary, "primary");

  std::cout << "\n-- Task *handoff = backup.release(): backup GIVES UP ownership; the caller must delete it --\n";
  Task *handoff = backup.release();
  peek(backup, "backup");
  std::cout << "  handoff->label = " << handoff->label << "\n";
  delete handoff;   // required -- release() bypassed unique_ptr's automatic cleanup

  std::cout << "\n-- primary.reset(): destroys primary's Task right now, without waiting for scope exit --\n";
  primary.reset();
  peek(primary, "primary");

  std::cout << "\n-- leaving main: both slots are already empty, so no further destruction happens --\n";
  return 0;
}
