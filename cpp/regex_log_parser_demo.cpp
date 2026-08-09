// regex_log_parser_demo.cpp
// Companion example for: topnotchnote.com/cpp/regular_expressions.html
//
// Parses a handful of log-style lines with one compiled-once pattern,
// pulling the date, level, and message out of each via capturing
// groups, then demonstrates regex_replace reformatting a date with
// backreferences.
// Compile:  g++ -Wall -Wextra -std=c++17 -o main regex_log_parser_demo.cpp

#include <iostream>
#include <regex>
#include <string>
#include <vector>

int main() {
  std::vector<std::string> lines = {
      "2026-08-09 INFO: build succeeded",
      "2026-08-08 ERROR: connection timed out",
      "not a log line at all",
  };

  // compiled once, reused for every line
  std::regex entry(R"((\d{4}-\d{2}-\d{2}) (\w+): (.+))");
  std::smatch m;

  std::cout << "-- parsing lines with regex_match --\n";
  for (const auto &line : lines) {
    if (std::regex_match(line, m, entry)) {
      std::cout << "  date=" << m[1] << "  level=" << m[2] << "  message=" << m[3] << "\n";
    } else {
      std::cout << "  (no match) " << line << "\n";
    }
  }

  std::cout << "\n-- regex_search finds a pattern anywhere, unlike regex_match --\n";
  std::regex digits(R"(\d+)");
  std::string mixed = "order #4471 shipped";
  if (std::regex_search(mixed, m, digits)) {
    std::cout << "  found digits: " << m[0] << " inside \"" << mixed << "\"\n";
  }

  std::cout << "\n-- regex_replace with backreferences --\n";
  std::regex iso(R"((\d{4})-(\d{2})-(\d{2}))");
  std::string date = "2026-08-09";
  std::string us = std::regex_replace(date, iso, "$2/$3/$1");
  std::cout << "  " << date << " -> " << us << "\n";

  return 0;
}
