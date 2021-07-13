#pragma once

#include <list>
#include <set>
#include <string>
#include <string_view>
#include <utility>

namespace search_server_storage_container {

class WordStorage {
public:
    void Insert(std::string word) {
        if (string_views_.count(word) == 0) {
            data_.push_back(std::move(word));
            string_views_.insert(data_.back());
        }
    }

    void Insert(std::string_view word) {
        if (string_views_.count(word) == 0) {
            data_.emplace_back(word);
            string_views_.insert(data_.back());
        }
    }

    std::set<std::string_view>::const_iterator Find(std::string_view word) {
        return string_views_.find(word);
    }

    auto end() {
        return string_views_.end();
    }

private:
    std::set<std::string_view> string_views_;
    std::list<std::string> data_;
};

}