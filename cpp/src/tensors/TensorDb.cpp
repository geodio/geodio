
#include "TensorDb.h"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <list>

namespace dio {

// Constructors

template<typename T>
TensorDb<T>::TensorDb() : cache_limit_(10) {}

template<typename T>
__attribute__((unused)) TensorDb<T>::TensorDb(const std::string& filename) : filename_(filename), cache_limit_(10) {
    load(filename);
}

template<typename T>
TensorDb<T>::TensorDb(const std::vector<std::pair<int, Tensor<T>>>& tensors) : cache_limit_(10) {
    for (const auto& pair : tensors) {
        add_tensor(pair.first, pair.second);
    }
}

// Destructor

template<typename T>
TensorDb<T>::~TensorDb() {
    if (!filename_.empty()) {
        save(filename_);
    }
}

// Database operations

template<typename T>
void TensorDb<T>::add_tensor(int id, const Tensor<T>& tensor) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Create metadata
    TensorMetadata metadata;
    metadata.id = id;
    metadata.data_type_size = sizeof(T);
    metadata.num_dimensions = tensor.shape().size();
    metadata.shape = tensor.shape();
    metadata.data_offset = 0;  // Will be updated during save

    // Update metadata map
    metadata_map_[id] = metadata;

    // Add tensor to cache
    tensor_cache_[id] = std::make_shared<Tensor<T>>(tensor);
    update_cache_order(id);
    evict_if_needed();
}

template<typename T>
void TensorDb<T>::remove_tensor(int id) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Remove from metadata and cache
    metadata_map_.erase(id);
    tensor_cache_.erase(id);
    cache_order_.remove(id);
}

template<typename T>
void TensorDb<T>::update_tensor(int id, const Tensor<T>& tensor) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (metadata_map_.find(id) == metadata_map_.end()) {
        throw std::runtime_error("Tensor ID not found.");
    }

    // Update metadata
    metadata_map_[id].num_dimensions = tensor.shape().size();
    metadata_map_[id].shape = tensor.shape();
    metadata_map_[id].data_type_size = sizeof(T);

    // Update tensor in cache
    tensor_cache_[id] = std::make_shared<Tensor<T>>(tensor);
    update_cache_order(id);
    evict_if_needed();
}

template<typename T>
std::shared_ptr<Tensor<T>> TensorDb<T>::get_tensor(int id) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if tensor is in cache
    auto it = tensor_cache_.find(id);
    if (it != tensor_cache_.end()) {
        update_cache_order(id);
        return it->second;
    }

    // Load tensor from file
    auto tensor = load_tensor(id);
    if (tensor) {
        tensor_cache_[id] = tensor;
        update_cache_order(id);
        evict_if_needed();
    }

    return tensor;
}

// File operations

template<typename T>
void TensorDb<T>::save(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string file_to_save = filename.empty() ? filename_ : filename;
    if (file_to_save.empty()) {
        throw std::runtime_error("No filename specified for saving.");
    }

    std::ofstream ofs(file_to_save, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file for saving.");
    }

    // Write placeholder for metadata size (will update later)
    size_t metadata_size = 0;
    ofs.write(reinterpret_cast<const char*>(&metadata_size), sizeof(size_t));

    // Collect tensor data offsets
    for (auto& pair : metadata_map_) {
        pair.second.data_offset = ofs.tellp();
        auto tensor_it = tensor_cache_.find(pair.first);
        if (tensor_it != tensor_cache_.end()) {
            write_tensor_to_file(ofs, pair.second, *tensor_it->second);
        } else {
            // Load tensor temporarily to write it
            auto tensor = load_tensor(pair.first);
            if (tensor) {
                write_tensor_to_file(ofs, pair.second, *tensor);
            } else {
                throw std::runtime_error("Tensor data missing for ID: " + std::to_string(pair.first));
            }
        }
    }

    // Save metadata position
    size_t metadata_pos = ofs.tellp();

    // Write metadata
    size_t num_tensors = metadata_map_.size();
    ofs.write(reinterpret_cast<const char*>(&num_tensors), sizeof(size_t));

    for (const auto& pair : metadata_map_) {
        const auto& metadata = pair.second;
        ofs.write(reinterpret_cast<const char*>(&metadata.id), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&metadata.data_type_size), sizeof(size_t));
        ofs.write(reinterpret_cast<const char*>(&metadata.num_dimensions), sizeof(size_t));

        for (size_t dim : metadata.shape) {
            ofs.write(reinterpret_cast<const char*>(&dim), sizeof(size_t));
        }

        ofs.write(reinterpret_cast<const char*>(&metadata.data_offset), sizeof(size_t));
    }

    // Update metadata size at the beginning of the file
    ofs.seekp(0, std::ios::beg);
    metadata_size = metadata_pos;
    ofs.write(reinterpret_cast<const char*>(&metadata_size), sizeof(size_t));

    ofs.close();
}

template<typename T>
void TensorDb<T>::load(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mutex_);

    filename_ = filename;
    std::ifstream ifs(filename_, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open file for loading.");
    }

    // Read metadata size
    size_t metadata_size = 0;
    ifs.read(reinterpret_cast<char*>(&metadata_size), sizeof(size_t));
    if (metadata_size == 0) {
        throw std::runtime_error("Invalid metadata size.");
    }

    // Seek to metadata position
    ifs.seekg(metadata_size, std::ios::beg);

    // Read number of tensors
    size_t num_tensors = 0;
    ifs.read(reinterpret_cast<char*>(&num_tensors), sizeof(size_t));

    for (size_t i = 0; i < num_tensors; ++i) {
        TensorMetadata metadata;
        ifs.read(reinterpret_cast<char*>(&metadata.id), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&metadata.data_type_size), sizeof(size_t));
        ifs.read(reinterpret_cast<char*>(&metadata.num_dimensions), sizeof(size_t));

        metadata.shape.resize(metadata.num_dimensions);
        for (size_t& dim : metadata.shape) {
            ifs.read(reinterpret_cast<char*>(&dim), sizeof(size_t));
        }

        ifs.read(reinterpret_cast<char*>(&metadata.data_offset), sizeof(size_t));

        metadata_map_[metadata.id] = metadata;
    }

    ifs.close();
}

// Cache management

template<typename T>
void TensorDb<T>::set_cache_limit(size_t max_tensors_in_cache) {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_limit_ = max_tensors_in_cache;
    evict_if_needed();
}

template<typename T>
size_t TensorDb<T>::get_cache_limit() const {
    return cache_limit_;
}

template<typename T>
void TensorDb<T>::evict_if_needed() {
    while (tensor_cache_.size() > cache_limit_) {
        // Evict least recently used tensor
        int id_to_evict = cache_order_.back();
        tensor_cache_.erase(id_to_evict);
        cache_order_.pop_back();
    }
}

template<typename T>
void TensorDb<T>::update_cache_order(int id) {
    cache_order_.remove(id);
    cache_order_.push_front(id);
}

// Helper methods

template<typename T>
void TensorDb<T>::write_tensor_to_file(std::ofstream& ofs, const TensorMetadata& metadata, const Tensor<T>& tensor) {
    // Write tensor data at current position
    size_t total_elements = tensor.compute_size(metadata.shape);
    ofs.write(reinterpret_cast<const char*>(tensor.get_data().data()), total_elements * sizeof(T));
}

template<typename T>
std::shared_ptr<Tensor<T>> TensorDb<T>::read_tensor_from_file(std::ifstream& ifs, const TensorMetadata& metadata) {
    // Seek to tensor data offset
    ifs.seekg(metadata.data_offset, std::ios::beg);

    // Read tensor data
    size_t total_elements = 1;
    for (size_t dim : metadata.shape) {
        total_elements *= dim;
    }

    std::vector<T> data(total_elements);
    ifs.read(reinterpret_cast<char*>(data.data()), total_elements * sizeof(T));

    // Create tensor
    auto tensor = std::make_shared<Tensor<T>>(data, metadata.shape);
    return tensor;
}

template<typename T>
std::shared_ptr<Tensor<T>> TensorDb<T>::load_tensor(int id) {
    auto it = metadata_map_.find(id);
    if (it == metadata_map_.end()) {
        throw std::runtime_error("Tensor ID not found in metadata.");
    }

    const auto& metadata = it->second;

    std::ifstream ifs(filename_, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open file for reading tensor.");
    }

    auto tensor = read_tensor_from_file(ifs, metadata);
    ifs.close();

    return tensor;
}

// Parallel loading

template<typename T>
std::future<std::shared_ptr<Tensor<T>>> TensorDb<T>::get_tensor_async(int id) {
    return std::async(std::launch::async, &TensorDb<T>::get_tensor, this, id);
}

}  // namespace dio

// Explicit template instantiation
template class dio::TensorDb<float>;
template class dio::TensorDb<double>;
template class dio::TensorDb<int>;
