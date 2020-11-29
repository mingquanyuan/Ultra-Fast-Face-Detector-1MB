// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the 
// specific language governing permissions and limitations under the License.

#pragma once
#include <fstream>
#include <chrono>
#include <stack>

#include "tnn/core/status.h"
using namespace std::chrono;

#define CHECK_API(status)                                                      \
    do {                                                                       \
        if (status != 0) {                                                     \
            fprintf(stderr, "API ERROR:%d\n", int(status));                    \
            return -1;                                                         \
        }                                                                      \
    } while (0)

#define CHECK_TNN_STATUS(status)                                               \
    do {                                                                       \
        if (status != TNN_NS::TNN_OK) {                                        \
            fprintf(stderr, "TNN API ERROR:0x%x", int(status));                \
            return status;                                                     \
        }                                                                      \
    } while (0)

// Helper functions
std::string fdLoadFile(std::string path) {
    std::ifstream file(path, std::ios::in);
    if (file.is_open()) {
        file.seekg(0, file.end);
        int size      = file.tellg();
        char* content = new char[size];
        file.seekg(0, file.beg);
        file.read(content, size);
        std::string fileContent;
        fileContent.assign(content, size);
        delete[] content;
        file.close();
        return fileContent;
    } else {
        return "";
    }
}

class Timer
{
public:
    std::stack<high_resolution_clock::time_point> tictoc_stack;

    void tic()
    {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        tictoc_stack.push(t1);
    }

    double toc(std::string msg = "", bool flag = true)
    {
        double diff = duration_cast<milliseconds>(high_resolution_clock::now() - tictoc_stack.top()).count();
        if(msg.size() > 0){
            if (flag)
                printf("%s time elapsed: %f ms\n", msg.c_str(), diff);
        }

        tictoc_stack.pop();
        return diff;
    }
    void reset()
    {
        tictoc_stack = std::stack<high_resolution_clock::time_point>();
    }
};