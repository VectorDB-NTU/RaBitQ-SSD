// Derived from Microsoft DiskANN (https://github.com/microsoft/DiskANN),
// include/concurrent_queue.h.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>

namespace rabitqlib
{

    // Thread-safe FIFO used as a buffer pool: workers pop a buffer, use it, and
    // push it back. pop() on an empty queue returns null_T rather than blocking,
    // so a caller that finds the pool empty waits on wait_for_push_notify() and
    // retries.
    template <typename T>
    class ConcurrentQueue
    {
        typedef std::chrono::microseconds chrono_us_t;
        typedef std::unique_lock<std::mutex> mutex_locker;

        std::queue<T> q;
        std::mutex mut;
        std::mutex push_mut;
        std::condition_variable push_cv;

    public:
        T null_T; // value returned by pop() when the queue is empty

        ConcurrentQueue()
        {
        }

        ~ConcurrentQueue()
        {
            this->push_cv.notify_all();
        }

        // PUSH BACK
        void push(const T &new_val)
        {
            mutex_locker lk(this->mut);
            this->q.push(new_val);
            lk.unlock();
        }

        // POP FRONT (returns null_T when empty)
        T pop()
        {
            mutex_locker lk(this->mut);
            if (this->q.empty())
            {
                lk.unlock();
                return this->null_T;
            }
            else
            {
                T ret = this->q.front();
                this->q.pop();
                lk.unlock();
                return ret;
            }
        }

        // register for notifications
        void wait_for_push_notify(chrono_us_t wait_time = chrono_us_t{10})
        {
            mutex_locker lk(this->push_mut);
            this->push_cv.wait_for(lk, wait_time);
            lk.unlock();
        }

        void push_notify_all()
        {
            this->push_cv.notify_all();
        }
    };
} // namespace rabitqlib
