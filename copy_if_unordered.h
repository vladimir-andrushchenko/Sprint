#pragma once

#include <vector>
#include <mutex>
#include <execution>
#include <algorithm>

namespace parallel_copy {

template <typename Container, typename Predicate>
std::vector<typename Container::value_type> CopyIfUnordered(const Container& container, Predicate predicate) {
    std::vector<typename Container::value_type> result;
    result.reserve(container.size());
    std::mutex result_mutex;
    std::for_each(
        std::execution::par,
        container.begin(), container.end(),
        [predicate, &result_mutex, &result](const auto& value) {
            if (predicate(value)) {
                typename Container::value_type* destination;
                {
                    std::lock_guard guard(result_mutex);
                    destination = &result.emplace_back();
                }
                *destination = value;
            }
        }
    );
    return result;
}

}