// lambda_map_filter_reduce.cpp
// Companion example for: topnotchnote.com/cpp/lambda_functional.html
//
// C++'s equivalent of map/filter/reduce, chained: std::transform,
// std::copy_if, and std::accumulate, each taking a lambda. Also shows
// a mutable lambda's captured-by-value copy diverging from the
// original variable back in the enclosing scope.
// Compile:  g++ -Wall -Wextra -std=c++17 -o main lambda_map_filter_reduce.cpp

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

void printAll(const std::string &label, const std::vector<int> &v) {
  std::cout << "  " << label << ":";
  for (int x : v) std::cout << " " << x;
  std::cout << "\n";
}

int main() {
  std::vector<int> nums{10, 11, 12, 13, 14, 15};
  printAll("nums", nums);

  std::vector<int> tripled;
  std::transform(nums.begin(), nums.end(), std::back_inserter(tripled),
                 [](int v) { return v * 3; });   // map
  printAll("tripled", tripled);

  std::vector<int> odds;
  std::copy_if(tripled.begin(), tripled.end(), std::back_inserter(odds),
               [](int v) { return v % 2 != 0; });   // filter
  printAll("odds", odds);

  int total = std::accumulate(odds.begin(), odds.end(), 0,
                               [](int acc, int v) { return acc + v; });   // reduce
  std::cout << "  sum of odds: " << total << "\n";

  std::cout << "\nmutable lambda vs. its enclosing-scope variable:\n";
  int counter = 0;
  auto inc = [counter]() mutable { return ++counter; };   // mutates the LAMBDA's own copy
  std::cout << "  inc() = " << inc() << "\n";
  std::cout << "  inc() = " << inc() << "\n";
  std::cout << "  outer counter is still: " << counter << "\n";

  return 0;
}
