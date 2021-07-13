#include <cassert>
#include <cmath>
#include <algorithm>
#include <execution>
#include <utility>
#include <numeric>

#include "search_server.h"
#include "string_processing.h"
#include "log_duration.h"

using namespace std::literals;

std::set<int>::const_iterator SearchServer::begin() const {
    return document_ids_.begin();
}

std::set<int>::const_iterator SearchServer::end() const {
    return document_ids_.end();
}

const std::map<std::string_view, double>& SearchServer::GetWordFrequencies(int document_id) const {
    const static std::map<std::string_view, double> empty_map;
    
    if (document_id_to_document_data_.count(document_id) > 0) {
        return document_id_to_document_data_.at(document_id).word_frequencies;
    }
    
    return empty_map;
}

void SearchServer::RemoveDocument(const int document_id) {
    RemoveDocument(std::execution::seq, document_id);
}

bool SearchServer::IsValidWord(const std::string_view word) const {
    // A valid word must not contain special characters
    return std::none_of(word.begin(), word.end(), [](auto c) {
        return c >= '\0' && c < ' ';
    });
}

SearchServer::SearchServer(const std::string& stop_words) {
    if (!IsValidWord(stop_words)) {
        throw std::invalid_argument("stop word contains unaccaptable symbol"s);
    }
    
    SetStopWords(stop_words);
}

SearchServer::SearchServer(const std::string_view stop_words) {
    if (!IsValidWord(stop_words)) {
        throw std::invalid_argument("stop word contains unaccaptable symbol"s);
    }
    
    SetStopWords(stop_words);
}

void SearchServer::SetStopWords(const std::string_view text) {
    for (const auto word : string_processing::SplitIntoWords(text)) {
        stop_words_.emplace(word);
    }
} // SetStopWords

bool SearchServer::AddDocument(int document_id, const std::string_view document,
                               DocumentStatus status, const std::vector<int>& ratings) {
    if (document_id < 0) {
        throw std::invalid_argument("negative ids are not allowed"s);
    }
    
    if (document_id_to_document_data_.count(document_id) > 0) {
        throw std::invalid_argument("repeating ids are not allowed"s);
    }
    
    if (!IsValidWord(document)) {
        throw std::invalid_argument("word in document contains unaccaptable symbol"s);
    }
    
    const std::vector<std::string_view> words = SplitIntoWordsNoStop(document);

    // save words to the server
    for (const auto word : words) {
        words_storage_.Insert(word);
    }
    
    const double inverse_word_count = 1.0 / static_cast<double>(words.size());
    
    std::map<std::string_view, double> word_frequencies;
    
    for (const std::string_view word : words) {
        // WordStorage has all the words that have been added to the search_server 
        const auto iterator_to_word_view_in_storage = words_storage_.Find(word);
        assert(iterator_to_word_view_in_storage != words_storage_.end());

        // use string views that store data in words_storage_ as keys
        word_to_document_id_to_term_frequency_[*iterator_to_word_view_in_storage][document_id] += inverse_word_count;
        word_frequencies[*iterator_to_word_view_in_storage] += inverse_word_count;
    }
    
    document_ids_.insert(document_id);
    
    document_id_to_document_data_.emplace(document_id, DocumentData{ComputeAverageRating(ratings), status, word_frequencies});
    
    return true; // this return is kind of redundant
} // AddDocument

int SearchServer::GetDocumentCount() const {
    return static_cast<int>(document_id_to_document_data_.size());
} // GetDocumentCount

std::vector<Document> SearchServer::FindTopDocuments(const std::string_view raw_query,
                                                     const DocumentStatus& desired_status) const {
    const auto predicate = [desired_status](int , DocumentStatus document_status, int ) {
        return document_status == desired_status;
    };
    
    return FindTopDocuments(std::execution::seq, raw_query, predicate);
} // FindTopDocuments with status as a second argument

std::tuple<std::vector<std::string_view>, DocumentStatus> SearchServer::MatchDocument(const std::string_view raw_query, int document_id) const {
   return MatchDocument(std::execution::seq, raw_query, document_id);
}

std::vector<std::string_view> SearchServer::SplitIntoWordsNoStop(const std::string_view text) const {
    std::vector<std::string_view> words;
    for (const std::string_view word : string_processing::SplitIntoWords(text)) {
        if (!IsStopWord(word)) {
            words.push_back(word);
        }
    }
    
    return words;
} // SplitIntoWordsNoStop

int SearchServer::ComputeAverageRating(const std::vector<int>& ratings) {
    int rating_sum = 0;
    
    for (const int rating : ratings) {
        rating_sum += rating;
    }
    
    return rating_sum / static_cast<int>(ratings.size());
} // ComputeAverageRating

bool SearchServer::IsStopWord(const std::string_view word) const {
    return stop_words_.count(word) > 0;
} // IsStopWord

SearchServer::QueryWord SearchServer::ParseQueryWord(std::string_view text) const {
    try {
        if (text.empty()) {
            throw std::invalid_argument("caught empty word, check for double spaces"s);
        }
    } catch(...) {
        exception_pointer_in_parse_query_word = std::current_exception();
    }    


    bool is_minus = false;
    
    try {
        if (text[0] == '-') {
            text = text.substr(1);
            
            if (text.empty()) {
                throw std::invalid_argument("empty minus words are not allowed"s);
            }
            
            if (text[0] == '-') {
                throw std::invalid_argument("double minus words are not allowed"s);
            }
            
            is_minus = true;
        }
        
        if (!IsValidWord(text)) {
            throw std::invalid_argument("special symbols in words are not allowed"s);
        }
    } catch(...) {
        exception_pointer_in_parse_query_word = std::current_exception();
    }
    
    return {text, is_minus, IsStopWord(text)};
} // ParseQueryWord

// Existence required
double SearchServer::ComputeWordInverseDocumentFrequency(const std::string_view word) const {
    assert(word_to_document_id_to_term_frequency_.count(word) != 0);
    
    const size_t number_of_documents_constains_word = word_to_document_id_to_term_frequency_.at(word).size();
    
    assert(number_of_documents_constains_word != 0);
    
    return std::log(static_cast<double>(GetDocumentCount()) / number_of_documents_constains_word);
} // ComputeWordInverseDocumentFrequency

namespace search_server_helpers {

void PrintMatchDocumentResult(int document_id, const std::vector<std::string_view> words, DocumentStatus status) {
    std::cout << "{ "s
    << "document_id = "s << document_id << ", "s
    << "status = "s << static_cast<int>(status) << ", "s
    << "words ="s;
    for (const std::string_view word : words) {
        std::cout << ' ' << word;
    }
    std::cout << "}"s << std::endl;
}

void AddDocument(SearchServer& search_server, int document_id, const std::string_view document, DocumentStatus status,
                 const std::vector<int>& ratings) {
    try {
        search_server.AddDocument(document_id, document, status, ratings);
    } catch (const std::exception& e) {
        std::cout << "Ошибка добавления документа "s << document_id << ": "s << e.what() << std::endl;
    }
}

void FindTopDocuments(const SearchServer& search_server, const std::string_view raw_query) {
    LOG_DURATION("Operation time");
    
    std::cout << "Результаты поиска по запросу: "s << raw_query << std::endl;
    
    try {
        for (const Document& document : search_server.FindTopDocuments(raw_query)) {
            PrintDocument(document);
        }
        
        std::cout << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Ошибка поиска: "s << e.what() << std::endl;
    }
}

void MatchDocuments(const SearchServer& search_server, const std::string_view query) {
    LOG_DURATION_STREAM("Operation time", std::cout);
    
    try {
        std::cout << "Матчинг документов по запросу: "s << query << std::endl;
        
        for (const int document_id : search_server) {
            const auto [words, status] = search_server.MatchDocument(query, document_id);
            
            PrintMatchDocumentResult(document_id, words, status);
        }
        
    } catch (const std::exception& e) {
        std::cout << "Ошибка матчинга документов на запрос "s << query << ": "s << e.what() << std::endl;
    }
}

SearchServer CreateSearchServer(const std::string_view stop_words) {
    SearchServer search_server;
    
    try {
        search_server = SearchServer(stop_words);
    } catch (const std::invalid_argument& e) {
        std::cout << "Ошибка создания search_server "s << ": "s << e.what() << std::endl;
    }
    
    return search_server;
}

} // namespace search_server_helpers
