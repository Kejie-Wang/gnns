//
// Created by WANG Kejie on 31/03/2017.
//

#ifndef GNNS_GENERAL_H
#define GNNS_GENERAL_H

#include <exception>

namespace gnns
{
    class GnnsException : public std::runtime_error
    {
    public:
        GnnsException(const char* message) : std::runtime_error(message) { }
        GnnsException(const std::string& message) : std::runtime_error(message) { }
    };
}

#endif //GNNS_GENERAL_H
