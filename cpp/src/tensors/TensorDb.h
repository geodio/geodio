//
// Created by zwartengaten on 9/16/24.
//

#ifndef GEODIO_TENSORDB_H
#define GEODIO_TENSORDB_H

#include "Tensor.h"
#include <unordered_map>
#include <mutex>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <condition_variable>
#include <queue>
#include <future>
#include <list>

namespace dio {

template<typename T>
class TensorDb {
public:
    // Constructors
    TensorDb();  // Empty database
    __attribute__((unused)) explicit TensorDb(const std::string& filename);  // Load from file
    explicit TensorDb(const std::vector<std::pair<int, Tensor<T>>>& tensors);  // Initialize from tensor list

    // Destructor
    ~TensorDb();

    // Database operations
    void add_tensor(int id, const Tensor<T>& tensor);
    void remove_tensor(int id);
    void update_tensor(int id, const Tensor<T>& tensor);
    std::shared_ptr<Tensor<T>> get_tensor(int id);

    // File operations
    void save(const std::string& filename = "");
    void load(const std::string& filename);

    // Cache management
    void set_cache_limit(size_t max_tensors_in_cache);
    [[nodiscard]] size_t get_cache_limit() const;

    // Parallel loading
    std::future<std::shared_ptr<Tensor<T>>> get_tensor_async(int id);

private:
    // Helper structures
    struct TensorMetadata {
        int id{};
        size_t data_type_size{};
        size_t num_dimensions{};
        std::vector<size_t> shape;
        size_t data_offset{};  // Offset in the file where data begins
    };

    // Member variables
    std::unordered_map<int, TensorMetadata> metadata_map_;  // Map from tensor ID to metadata
    std::unordered_map<int, std::shared_ptr<Tensor<T>>> tensor_cache_;  // Cached tensors
    std::list<int> cache_order_{};  // For cache eviction (least recently used)
    size_t cache_limit_;  // Maximum number of tensors to keep in cache
    std::mutex mutex_;  // Mutex for thread safety
    std::string filename_;  // Database file name

    // Helper methods
//    void load_metadata();
//    void save_metadata();
    void evict_if_needed();
    void update_cache_order(int id);

    // File I/O methods
    void write_tensor_to_file(std::ofstream& ofs, const TensorMetadata& metadata, const Tensor<T>& tensor);
    std::shared_ptr<Tensor<T>> read_tensor_from_file(std::ifstream& ifs, const TensorMetadata& metadata);

    // Parallel loading helpers
    std::shared_ptr<Tensor<T>> load_tensor(int id);
};

}  // namespace dio

#endif //GEODIO_TENSORDB_H
