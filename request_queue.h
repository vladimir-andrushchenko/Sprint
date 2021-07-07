#pragma once

#include <deque>
#include <vector>
#include <string>

#include "document.h"
#include "search_server.h"

class RequestQueue {
public:
    explicit RequestQueue(const SearchServer& search_server);
    
public:
    template <typename DocumentPredicate>
    std::vector<Document> AddFindRequest(const std::string& raw_query, DocumentPredicate document_predicate);
    
    std::vector<Document> AddFindRequest(const std::string& raw_query,
                                         DocumentStatus status = DocumentStatus::ACTUAL);
    
    int GetNoResultRequests() const;
    
private:
    struct QueryResult {
        QueryResult(const std::string& raw_query, int results): raw_query(raw_query), results(results) {}
        
        const std::string raw_query;
        const int results;
    };
    
private:
    static constexpr int kMinutesInADay = 1440;
    
private:
    void RemoveOutdatedRequests();
    
    void RecordRequest(const std::string& raw_query, int results);
    
private:
    std::deque<QueryResult> requests_;
    const SearchServer& server_;
    int no_result_requests_counter_ = 0;
};

template <typename DocumentPredicate>
std::vector<Document> RequestQueue::AddFindRequest(const std::string& raw_query, DocumentPredicate document_predicate) {
    RemoveOutdatedRequests();
    
    const std::vector<Document>& results = server_.FindTopDocuments(raw_query, document_predicate);
    
    RecordRequest(raw_query, static_cast<int>(results.size()));
    
    return results;
}
