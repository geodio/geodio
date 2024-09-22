//
// Created by zwartengaten on 9/22/24.
//

#ifndef GEODIO_SLICE_H
#define GEODIO_SLICE_H

namespace dio {

    class Slice {
    public:
        Slice(int start, int end, int step = 1)
            : start_(start), end_(end), step_(step) {}

        explicit Slice(int start)
            : start_(start), end_(start + 1), step_(1) {}


        [[nodiscard]] int start() const { return start_; }
        [[nodiscard]] int end() const { return end_; }
        [[nodiscard]] int step() const { return step_; }

    private:
        int start_, end_, step_;
    };

} // dio

#endif //GEODIO_SLICE_H
