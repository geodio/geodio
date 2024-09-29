//    Geodio Neural Network Framework - A lightweight neural network library focused on custom architecture design and optimization.
//    Copyright (C) Copyright (C) 2024 Geodio (created by Rareș Polenciuc)
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#ifndef GEODIO_SLICE_H
#define GEODIO_SLICE_H
#include <optional>

namespace dio {



    class Slice {

    public:
        Slice(std::optional<int> start, std::optional<int> end, int step = 1)
            : start_(start), end_(end), step_(step) {}

        explicit Slice(std::optional<int> start)
            : start_(start), end_(start ? std::optional<int>(*start + 1) : std::nullopt), step_(1) {}

        // Accessors with proper handling of optional values
        [[nodiscard]] std::optional<int> start() const { return start_; }

        [[nodiscard]] std::optional<int> end() const { return end_; }

        [[nodiscard]] int step() const { return step_; }

        // Method to resolve negative or empty indices based on tensor size
        [[nodiscard]] int resolve_start(int dimension_size) const {
            if (!start_) return 0;  // Default to 0 if empty
            return (*start_ < 0) ? dimension_size + *start_ : *start_;  // Adjust for negative indices
        }

        [[nodiscard]] int resolve_end(int dimension_size) const {
            if (!end_) return dimension_size;  // Default to dimension size if empty
            return (*end_ < 0) ? dimension_size + *end_ : *end_;  // Adjust for negative indices
        }

    private:
        std::optional<int> start_, end_;
        int step_;
    };



} // namespace dio



#endif //GEODIO_SLICE_H
