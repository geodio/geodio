//
// Created by zwartengaten on 9/30/24.
//

#ifndef GEODIO_EXECUTIONCONTEXT_H
#define GEODIO_EXECUTIONCONTEXT_H

#include <unordered_map>
#include <vector>
#include "AnyTensor.h"
#include "ComputationalGraph.h"

namespace dio {

    enum class CachingMode {
        CacheAll,      // Cache everything (needed for backpropagation through time)
        CacheMinimal   // Cache only what's necessary and clean up when data leaves context
    };
    using CACHE = std::unordered_map<int, std::unordered_map<int, a_tens>>;

    class ExecutionContext {
    public:
        ComputationalGraph* graph;
        CACHE operand_cache;  // time_step -> operand_id -> result
        std::unordered_map<int, std::unordered_map<int, bool>> execution_path;  // For conditionals
        int current_time_step = 0;
        std::vector<a_tens> args;
        CachingMode caching_mode;
        ExecutionContext* parent_context;

        ExecutionContext(ComputationalGraph* g, const std::vector<a_tens>& a, CachingMode mode,
                         ExecutionContext * parent_context = nullptr)
            : graph(g), args(a), caching_mode(mode), parent_context(parent_context) {}

        // Get the result of an operand from the cache
        a_tens get(int operand_id);

        // Set the result of an operand in the cache
        void set(int operand_id, const a_tens& result);

        bool has_operand_result(int operand_id);

        // Clean the cache (used in CacheMinimal mode)
        void clean_cache();

        Operand get_op(int op_id) const;
    };


} // dio

#endif //GEODIO_EXECUTIONCONTEXT_H
