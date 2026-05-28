#pragma once

#include <Arduino.h>

namespace uv_test {

template <typename T, size_t Capacity>
class RingBuffer {
 public:
  bool push(const T& value) {
    if (size_ == Capacity) {
      return false;
    }

    storage_[tail_] = value;
    tail_ = (tail_ + 1) % Capacity;
    ++size_;
    return true;
  }

  bool pop(T& value) {
    if (size_ == 0) {
      return false;
    }

    value = storage_[head_];
    head_ = (head_ + 1) % Capacity;
    --size_;
    return true;
  }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

 private:
  T storage_[Capacity];
  size_t head_ = 0;
  size_t tail_ = 0;
  size_t size_ = 0;
};

}  // namespace uv_test
