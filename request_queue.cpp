#include "request_queue.h"

RequestQueue::RequestQueue(const SearchServer& search_server): server_(search_server) {}

std::vector<Document> RequestQueue::AddFindRequest(const std::string& raw_query,
                                     DocumentStatus status)  {
    return AddFindRequest(raw_query, [&status](int, DocumentStatus doc_status, int){
        return doc_status == status;
    });
}

int RequestQueue::GetNoResultRequests() const {
    return no_result_requests_counter_;
}

void RequestQueue::RemoveOutdatedRequests() {
    if (requests_.size() >= kMinutesInADay) {
        if(requests_.front().results == 0) {
            --no_result_requests_counter_;
        }
        
        requests_.pop_front();
    }
}

void RequestQueue::RecordRequest(const std::string& raw_query, int results) {
    requests_.push_back(QueryResult(raw_query, results));
    
    if (results == 0) {
        ++no_result_requests_counter_;
    }
}
