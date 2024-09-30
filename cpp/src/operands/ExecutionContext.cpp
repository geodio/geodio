//
// Created by zwartengaten on 9/30/24.
//

#include "ExecutionContext.h"

namespace dio {
    a_tens ExecutionContext::get(int operand_id) {
        int time_step = (caching_mode == CachingMode::CacheAll) ? current_time_step : 0;
        if (operand_cache[time_step].find(operand_id) != operand_cache[time_step].end()) {
            return operand_cache[time_step][operand_id];
        } else if (parent_context != nullptr) {
            return parent_context->get(operand_id);
        } else {
            throw std::runtime_error("Operand result not found in any context");
        }
    }

    void ExecutionContext::set(int operand_id, const a_tens &result) {
        int time_step = (caching_mode == CachingMode::CacheAll) ? current_time_step : 0;
        operand_cache[time_step][operand_id] = result;
    }

    void ExecutionContext::clean_cache() {
        if (caching_mode == CachingMode::CacheMinimal) {
            operand_cache.clear();
        }
    }

    bool ExecutionContext::has_operand_result(int operand_id) {
        int time_step = (caching_mode == CachingMode::CacheAll) ? current_time_step : 0;
        if (operand_cache[time_step].find(operand_id) != operand_cache[time_step].end()) {
            return true;
        } else if (parent_context != nullptr) {
            return parent_context->has_operand_result(operand_id);
        } else {
            return false;
        }
    }

    Operand ExecutionContext::get_op(int op_id) const{
        return graph->operands[op_id];
    }
} // dio