// stl_containers_custom_comparator_map.cpp
// Companion example for: topnotchnote.com/cpp/stl_containers.html
//
// A std::map ordered by a custom comparator lambda (absolute value
// instead of the default <), plus the three equivalent ways existing
// code inserts into a map -- operator[], insert() with make_pair, and
// insert() with brace initialization.
// Compile:  g++ -Wall -Wextra -std=c++17 -o main stl_containers_custom_comparator_map.cpp

#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

int main() {
  std::cout << "-- map with a custom comparator (order by absolute value) --\n";
  auto byAbsValue = [](int a, int b) { return std::abs(a) < std::abs(b); };
  std::map<int, int, decltype(byAbsValue)> ordered(byAbsValue);

  std::vector<int> values = {1, -2, 3, 4, 2};
  for (int v : values) ordered.insert({v, v});

  std::cout << "  keys in order:";
  for (auto &kv : ordered) std::cout << " " << kv.first;
  std::cout << "  (2 was a duplicate key by absolute value, so it's dropped)\n";

  std::cout << "\n-- three ways to insert into a map --\n";
  std::map<std::string, int> inventory = {{"espresso", 20}, {"latte", 8}};
  inventory["mocha"] = 5;                            // operator[] -- inserts if absent, overwrites if present
  inventory.insert(std::make_pair("chai", 12));        // insert() -- does NOT overwrite an existing key
  inventory.insert({"cortado", 3});                     // insert() with brace initialization

  inventory["espresso"] = 99;   // overwrite via operator[]
  auto notOverwritten = inventory.insert(std::make_pair("latte", 999));   // insert() refuses to overwrite
  std::cout << "  insert() on existing key \"latte\" succeeded? "
            << (notOverwritten.second ? "true" : "false") << " -- still " << inventory["latte"] << "\n";

  std::cout << "  final inventory:\n";
  for (auto &kv : inventory) std::cout << "    " << kv.first << " -> " << kv.second << "\n";

  return 0;
}
