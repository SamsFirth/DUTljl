
#include "task_queue.h"

TaskQueue::TaskQueue() {
    worker = std::thread(&TaskQueue::processTasks, this);
    sync_flag.store(true, std::memory_order_seq_cst);
    exit_flag.store(false, std::memory_order_seq_cst);
}

TaskQueue::~TaskQueue() {
    {
        mutex.lock();
        exit_flag.store(true, std::memory_order_seq_cst);
        mutex.unlock();
    }
    cv.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
}

void TaskQueue::enqueue(std::function<void()> task) {
    {
        mutex.lock();
        tasks.push(task);
        sync_flag.store(false, std::memory_order_seq_cst);
        mutex.unlock();
    }
    cv.notify_one();
}

void TaskQueue::sync() {
    while (!sync_flag.load(std::memory_order_seq_cst))
        ;
}

void TaskQueue::processTasks() {
    while (true) {
        std::function<void()> task;
        {
            mutex.lock();
            cv.wait(mutex, [this]() { return !tasks.empty() || exit_flag.load(std::memory_order_seq_cst); });
            if (exit_flag.load(std::memory_order_seq_cst) && tasks.empty()) {
                return;
            }
            task = tasks.front();
            tasks.pop();
            mutex.unlock();
        }
        task();
        {
            mutex.lock();
            if (tasks.empty()) {
                sync_flag.store(true, std::memory_order_seq_cst);
            }
            mutex.unlock();
        }
    }
}