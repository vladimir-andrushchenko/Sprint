#include "read_input_functions.h"

#include <iostream>

namespace read_input {

std::string ReadLine() {
    std::string line;
    getline(std::cin, line);
    return line;
}

int ReadLineWithNumber() {
    int result;
    std::cin >> result;
    ReadLine();
    return result;
}

} // namespace read_input
