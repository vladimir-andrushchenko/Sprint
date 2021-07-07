#pragma once

#include <iostream>

struct Document {
    int id = 0;
    double relevance = 0.0;
    int rating = 0;
    
    Document();
    Document(int id, double relevance, int rating);
};

enum class DocumentStatus {
    ACTUAL,
    IRRELEVANT,
    BANNED,
    REMOVED,
};

std::ostream& operator<<(std::ostream& out, const DocumentStatus status);

void PrintDocument(const Document& document);

std::ostream& operator<<(std::ostream& output, const Document& document);

