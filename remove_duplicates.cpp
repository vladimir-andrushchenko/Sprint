#include "remove_duplicates.h"

#include <set>
#include <string>
#include <iostream>
#include <vector>

using namespace std::literals;

namespace remove_duplicates {

void RemoveDuplicates(SearchServer& search_server) {
    std::set<std::set<std::string>> unique_documents;
    
    std::vector<int> duplicate_document_ids;
    
    for (const int document_id : search_server) {
        const auto words_to_term_frequencies = search_server.GetWordFrequencies(document_id);
        
        std::set<std::string> words_in_document;
        
        for (const auto& [word, term_frequency] : words_to_term_frequencies) {
            words_in_document.insert(word);
        }
        
        if (!unique_documents.insert(words_in_document).second) {
            duplicate_document_ids.push_back(document_id);
            std::cout << "Found duplicate document id "s << document_id << std::endl;
        }
    }
    
    for (const int duplicate_id : duplicate_document_ids) {
        search_server.RemoveDocument(duplicate_id);
    }
}

}
