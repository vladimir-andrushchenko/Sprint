#pragma once

#include <algorithm>
#include <deque>
#include <future>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace std::string_literals;

template <typename K, typename V>
class ConcurrentMap {
public:
    static_assert(std::is_integral_v<K>,
                  "ConcurrentMap supports only integer keys");

    struct Access {
        std::lock_guard<std::mutex> guard;
        V &ref_to_value;
    };

    explicit ConcurrentMap(size_t bucket_count) : BUCKET_COUNT(bucket_count) {
        all_maps.resize(BUCKET_COUNT);
    }

    Access operator[](const K &key) {
        size_t map_key = key % BUCKET_COUNT;
        auto &curr_map = all_maps[map_key];
        return Access{std::lock_guard(curr_map.bucket_mutex_),
                      curr_map.bucket_map_[key]};
    }

    std::map<K, V> BuildOrdinaryMap() {
        std::map<K, V> result;
        for (auto &map : all_maps) {
            for (const auto &[key, value] : map.bucket_map_) {
                result[key] = operator[](key).ref_to_value;
            }
        }
        return result;
    }

    int GetSize() {
        int result{};

        for (const auto& bucket : all_maps) {
            result += bucket.bucket_map_.size();
        }

        return result;
    }

    size_t getBucketCount() {
        return BUCKET_COUNT;
    }

private:
    const size_t BUCKET_COUNT;

    struct Bucket {
        std::map<K, V> bucket_map_;
        std::mutex bucket_mutex_;
    };

    std::deque<Bucket> all_maps;
};
