#pragma once

#include <vector>
#include <string>
#include <algorithm>

using namespace std::literals;

template<typename InputIterator>
class IteratorRange {
public:
    IteratorRange(InputIterator range_begin, InputIterator range_end): begin_iterator_(range_begin), end_iterator_(range_end) {}
    
public:
    InputIterator begin() const {
        return begin_iterator_;
    }
    
    InputIterator end() const {
        return end_iterator_;
    }
    
    size_t size() const {
        return std::distance(begin_iterator_, end_iterator_);
    }
    
private:
    InputIterator begin_iterator_;
    InputIterator end_iterator_;
};

template <typename Iterator>
class Paginator {
public:
    Paginator() = default;

public:
    void Init(Iterator range_begin, Iterator range_end, size_t page_size) {
        if (range_begin == range_end) {
            throw std::invalid_argument("no empty ranges"s);
        }

        size_t left = std::distance(range_begin, range_end);

        while (left > 0) {
            const auto current_page_size = std::min(page_size, left);
            pages_.push_back({range_begin, range_begin + current_page_size});
            left -= current_page_size;
            range_begin += current_page_size;
        }
    }
    
public:
    bool IsInitialized() {
        return !(pages_.begin() == pages_.end());
    }
    
    auto begin() const {
        return pages_.begin();
    }
    
    auto end() const {
        return pages_.end();
    }
    
    size_t Size() {
        return pages_.size();
    }
    
private:
    std::vector<IteratorRange<Iterator>> pages_;
};

template <typename Container>
auto Paginate(const Container& container, size_t page_size) {
    Paginator<decltype(std::begin(container))> paginator;
    paginator.Init(std::begin(container), std::end(container), page_size);
    return paginator;
}

template<typename IteratorRangeType>
std::ostream& operator<<(std::ostream& output, const IteratorRange<IteratorRangeType>& iterator_range) {
    for (auto it = iterator_range.begin(); it != iterator_range.end(); ++it) {
        output << *it;
    }
    return output;
}
