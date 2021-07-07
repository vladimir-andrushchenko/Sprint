#pragma once

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <sstream>

using namespace std::string_literals;

// logging functionality for containers
// log pair
template<typename First, typename Second>
std::ostream& operator<<(std::ostream& out, const std::pair<First, Second>& container) {
    out << container.first << ": "s << container.second;
    
    return out;
}

template<typename Container>
void Print(std::ostream& out, const Container& container) {
    bool isFirst = true;
    
    for (const auto& element : container) {
        if(isFirst) {
            out << element;
            
            isFirst = false;
            
            continue;
        }
        
        out << ", "s << element;
    }
}

// log vector
template<typename Element>
std::ostream& operator<<(std::ostream& output, const std::vector<Element>& container) {
    output << '[';
    
    Print(output, container);
    
    output << ']';
    
    return output;
}

// log set
template<typename Element>
std::ostream& operator<<(std::ostream& output, const std::set<Element>& container) {
    output << '{';
    
    Print(output, container);
    
    output << '}';
    
    return output;
}

// log map
template<typename Key, typename Value>
std::ostream& operator<<(std::ostream& out, const std::map<Key, Value>& container) {
    out << '{';
    
    Print(out, container);
    
    out << '}';
    
    return out;
}

// testing framework
template <typename TestFunction>
void RunTestImplementation(TestFunction test_function, const std::string& function_name) {
    test_function();
    
    std::cerr << function_name << " OK\n"s;
}

template <typename T, typename U>
void AssertEqualImplementation(const T& t, const U& u, const std::string& t_str, const std::string& u_str,
                               const std::string& file, const std::string& func, unsigned line, const std::string& hint) {
    if (t != u) {
        std::cerr << std::boolalpha;
        
        std::cerr << file << "("s << line << "): "s << func << ": "s;
        
        std::cerr << "ASSERT_EQUAL("s << t_str << ", "s << u_str << ") failed: "s;
        
        std::cerr << t << " != "s << u << "."s;
        
        if (!hint.empty()) {
            std::cerr << " Hint: "s << hint;
        }
        
        std::cerr << std::endl;
        
        abort();
    }
}

void AssertImplementation(bool value, const std::string& expr_str, const std::string& file,
                          const std::string& func, unsigned line, const std::string& hint) {
    if (!value) {
        std::cerr << file << "("s << line << "): "s << func << ": "s;
        
        std::cerr << "ASSERT("s << expr_str << ") failed."s;
        
        if (!hint.empty()) {
            std::cerr << " Hint: "s << hint;
        }
        
        std::cerr << std::endl;
        
        abort();
    }
}

#define ASSERT_EQUAL(a, b) AssertEqualImplementation((a), (b), #a, #b, __FILE__, __FUNCTION__, __LINE__, ""s)

#define ASSERT_EQUAL_HINT(a, b, hint) AssertEqualImplementation((a), (b), #a, #b, __FILE__, __FUNCTION__, __LINE__, (hint))

#define ASSERT(expression) AssertImplementation((expression), #expression, __FILE__, __FUNCTION__, __LINE__, ""s)

#define ASSERT_HINT(expression, hint) AssertImplementation((expression), #expression, __FILE__, __FUNCTION__, __LINE__, (hint))

#define RUN_TEST(test_function) RunTestImplementation((test_function), #test_function)

