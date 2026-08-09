// stl_algorithms_partition_nth_element.cpp
// Companion example for: topnotchnote.com/cpp/stl_algorithms_iterators.html
//
// std::stable_partition (group by predicate, preserve relative order)
// and std::nth_element (k-th smallest without a full sort) run
// side by side so you can see exactly what each one guarantees about
// the resulting order -- and what it doesn't.
// Compile:  g++ -Wall -Wextra -std=c++17 -o main stl_algorithms_partition_nth_element.cpp

#include <algorithm>
#include <iostream>
#include <vector>

void printAll(const std::string &label, const std::vector<int> &v) {
  std::cout << "  " << label << ":";
  for (int x : v) std::cout << " " << x;
  std::cout << "\n";
}

int main() {
  std::vector<int> nums = {1, 2, 3, 4, 5, 6, 7, 8};
  printAll("before", nums);

  auto boundary = std::stable_partition(nums.begin(), nums.end(),
                                         [](int i) { return i % 2 != 0; });
  printAll("after stable_partition (odds, then evens)", nums);
  std::cout << "  first even is at index " << (boundary - nums.begin()) << "\n";

  std::vector<int> v = {1, 5, 4, 2, 9, 7, 3, 8, 2};
  printAll("\n  before nth_element", v);
  std::nth_element(v.begin(), v.begin() + 4, v.end());
  printAll("after nth_element(begin+4)", v);
  std::cout << "  v[4] = " << v[4]
            << " -- guaranteed to be what it would be after a full sort;"
               " neither side is fully ordered\n";

  return 0;
}
