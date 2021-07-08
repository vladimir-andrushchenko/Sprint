#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <execution>
#include <list>

#include "document.h"
#include "string_processing.h"

using namespace std::literals;

class SearchServer {
public:
    SearchServer() = default;
    
    template <typename StringCollection>
    explicit SearchServer(const StringCollection& stop_words);
    
    explicit SearchServer(const std::string_view stop_words);

    explicit SearchServer(const std::string& stop_words);
    
public:
    void SetStopWords(const std::string_view text);
    
    bool AddDocument(int document_id, const std::string_view document,
                     DocumentStatus status, const std::vector<int>& ratings);
    
    int GetDocumentCount() const;
    
    template<typename Predicate>
    std::vector<Document> FindTopDocuments(const std::string& raw_query, Predicate predicate) const;
    
    std::vector<Document> FindTopDocuments(const std::string& raw_query,
                                           const DocumentStatus& desired_status = DocumentStatus::ACTUAL) const;
    
    std::tuple<std::vector<std::string>, DocumentStatus> MatchDocument(const std::string& raw_query, const int document_id) const;

    template<typename ExecutionPolicy>
    std::tuple<std::vector<std::string>, DocumentStatus> MatchDocument(const ExecutionPolicy& policy, const std::string& raw_query, const int document_id) const;
    
    std::set<int>::const_iterator begin() const;
    
    std::set<int>::const_iterator end() const;
    
    const std::map<std::string, double>& GetWordFrequencies(int document_id) const;
    
    void RemoveDocument(const int document_id);

    template<typename ExecutionPolicy>
    void RemoveDocument(const ExecutionPolicy& p, const int document_id);

private:
    struct DocumentData {
        int rating = 0;
        DocumentStatus status = DocumentStatus::ACTUAL;
        std::map<std::string, double> word_frequencies;
    };
    
    struct Query {
        std::set<std::string> plus_words;
        std::set<std::string> minus_words;

        Query& operator+=(Query&& other) {
            for (const auto& other_plus_word : other.plus_words) {
                plus_words.insert(std::move(other_plus_word));
            }

            for (const auto& other_minus_word : other.minus_words) {
                minus_words.insert(std::move(other_minus_word));
            }

            return *this;
        }
    };
    
    struct QueryWord {
        std::string data;
        bool is_minus = false;
        bool is_stop = false;
    };
    
private:
    static constexpr int kMaxResultDocumentCount = 5;
    static constexpr double kAccuracy = 1e-6;
    
private:
    std::vector<std::string> SplitIntoWordsNoStop(const std::string& text) const;
    
    static int ComputeAverageRating(const std::vector<int>& ratings);
    
    bool IsStopWord(const std::string& word) const;
    
    QueryWord ParseQueryWord(std::string text) const;

    template<typename ExecutionPolicy>
    Query ParseQuery(const ExecutionPolicy& p, const std::string& text) const;
    
    // Existence required
    double ComputeWordInverseDocumentFrequency(const std::string& word) const;
    
    std::vector<Document> FindAllDocuments(const Query& query) const;

    bool IsValidWord(const std::string_view word) const;
    
private:
    std::set<std::string> stop_words_;

    std::list<std::string> words_storage_;
    
    std::map<std::string, std::map<int, double>> word_to_document_id_to_term_frequency_;
    
    std::map<int, DocumentData> document_id_to_document_data_;
    
    std::set<int> document_ids_;
};

template<typename ExecutionPolicy>
SearchServer::Query SearchServer::ParseQuery(const ExecutionPolicy& policy, const std::string& text) const {
    // this is a temporary workaround and should be removed
    if (text.find("--") != text.npos || !IsValidWord(text)) {
        throw std::invalid_argument("omg\n");
    }

    auto words = string_processing::SplitIntoWords(text);

    // this is a temporary workaround and should be removed
    if (std::find(words.begin(), words.end(), "-"s) != words.end()) {
        throw std::invalid_argument("omg2\n");
    }

    // UnaryOp
    const auto transform_word_in_query = [this](const std::string& word){
        auto query_word = this->ParseQueryWord(word);

        Query query;
        if (!query_word.is_stop) {
            if (query_word.is_minus) {
                query.minus_words.insert(query_word.data);
            } else {
                query.plus_words.insert(query_word.data);
            }
        }

        return query;
    };

    // BinaryOp
    const auto combine_queries = [](Query first, Query second){
        return first += std::move(second);
    };

    return std::transform_reduce(policy, std::make_move_iterator(words.begin()), std::make_move_iterator(words.end()), Query{}, combine_queries, transform_word_in_query);
} // ParseQuery

template<typename ExecutionPolicy>
std::tuple<std::vector<std::string>, DocumentStatus> SearchServer::MatchDocument(const ExecutionPolicy& policy, const std::string& raw_query, int document_id) const {
    const Query query = ParseQuery(policy, raw_query);
    
    std::vector<std::string> matched_words;
    for (const std::string& word : query.plus_words) {
        if (word_to_document_id_to_term_frequency_.count(word) == 0) {
            continue;
        }
        
        if (word_to_document_id_to_term_frequency_.at(word).count(document_id)) {
            matched_words.push_back(word);
        }
    }
    
    for (const std::string& word : query.minus_words) {
        if (word_to_document_id_to_term_frequency_.count(word) == 0) {
            continue;
        }
        
        if (word_to_document_id_to_term_frequency_.at(word).count(document_id)) {
            matched_words.clear();
            break;
        }
    }
    
    return std::tuple<std::vector<std::string>, DocumentStatus>{matched_words, document_id_to_document_data_.at(document_id).status};
} // MatchDocument

template<typename ExecutionPolicy>
void SearchServer::RemoveDocument(const ExecutionPolicy& policy, const int document_id) {
    if (document_id_to_document_data_.count(document_id) == 0) {
        return;
    }

    // get list of words that are in this doc
    const auto words_and_frequencies = GetWordFrequencies(document_id);

    // initialize linear container that will contain inner maps of word_to_document_id_to_term_frequency where id points to frequency
    std::vector<std::map<int, double>> id_to_frequency;
    id_to_frequency.reserve(words_and_frequencies.size());

    // populate this inner container
    for (const auto& [word, term_frequency] : words_and_frequencies) {
        id_to_frequency.push_back(std::move(word_to_document_id_to_term_frequency_.at(word)));
    }

    // change inner maps
    std::for_each(policy, id_to_frequency.begin(), id_to_frequency.end(), [document_id](std::map<int, double>& element){
        element.erase(document_id);
    });

    // and put them back
    auto iterator_for_id_to_frequency_maps = id_to_frequency.begin();
    for (const auto& [word, term_frequency] : words_and_frequencies) {
        word_to_document_id_to_term_frequency_.at(word) = std::move(*(iterator_for_id_to_frequency_maps++));
        
        if (word_to_document_id_to_term_frequency_.at(word).empty()) {
            word_to_document_id_to_term_frequency_.erase(word);
        }
    }

    // not parallel
    document_id_to_document_data_.erase(document_id);
    
    document_ids_.erase(document_id);
}

template <typename StringCollection>
SearchServer::SearchServer(const StringCollection& stop_words) {
    using namespace std::literals;
    
    for (const auto& stop_word : stop_words) {
        if (!IsValidWord(stop_word)) {
            throw std::invalid_argument("stop word contains unaccaptable symbol"s);
        }
        
        stop_words_.emplace(stop_word);
    }
}

template<typename Predicate>
std::vector<Document> SearchServer::FindTopDocuments(const std::string& raw_query, Predicate predicate) const {
    const Query query = ParseQuery(std::execution::seq, raw_query);
    
    std::vector<Document> matched_documents = FindAllDocuments(query);
    
    std::vector<Document> filtered_documents;
    for (const Document& document : matched_documents) {
        const auto document_status = document_id_to_document_data_.at(document.id).status;
        const auto document_rating = document_id_to_document_data_.at(document.id).rating;
        
        if (predicate(document.id, document_status, document_rating)) {
            filtered_documents.push_back(document);
        }
    }
    
    std::sort(filtered_documents.begin(), filtered_documents.end(),
              [](const Document& left, const Document& right) {
        if (std::abs(left.relevance - right.relevance) < kAccuracy) {
            return left.rating > right.rating;
        } else {
            return left.relevance > right.relevance;
        }
    });
    
    if (static_cast<int>(filtered_documents.size()) > kMaxResultDocumentCount) {
        filtered_documents.resize(static_cast<size_t>(kMaxResultDocumentCount));
    }
    
    return filtered_documents;
} // FindTopDocuments 

namespace search_server_helpers {

void PrintMatchDocumentResult(int document_id, const std::vector<std::string>& words, DocumentStatus status);

void AddDocument(SearchServer& search_server, int document_id, const std::string& document, DocumentStatus status,
                 const std::vector<int>& ratings);

void FindTopDocuments(const SearchServer& search_server, const std::string& raw_query);

void MatchDocuments(const SearchServer& search_server, const std::string& query);

SearchServer CreateSearchServer(const std::string& stop_words);

} // namespace search_server_helpers



