/**
 * @file thread_pool.h
 * @brief Simple thread pool for parallel task execution
 * 
 * Thread pools avoid the overhead of creating/destroying threads for each task.
 * Instead, threads are created once and wait for work to do.
 * 
 * ANALOGY: A restaurant kitchen
 * - Without thread pool: Hire a new chef for each order, fire after (expensive!)
 * - With thread pool: Keep 4 chefs always working, assign orders as they come
 */

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <stddef.h>

/**
 * @brief Opaque thread pool structure
 */
typedef struct ThreadPool ThreadPool;

/**
 * @brief Get the number of threads in the pool
 * @param pool Thread pool
 * @return Number of worker threads
 */
int thread_pool_get_num_threads(ThreadPool* pool);

/**
 * @brief Create a thread pool with specified number of worker threads
 * @param num_threads Number of worker threads to create
 * @return Thread pool handle, or NULL on error
 * 
 * Creates a pool of persistent worker threads that wait for work.
 */
ThreadPool* thread_pool_create(int num_threads);

/**
 * @brief Submit a task to the thread pool
 * @param pool Thread pool
 * @param fn Function to execute (takes void* argument, returns void*)
 * @param arg Argument to pass to function
 * @return 0 on success, -1 on error
 * 
 * The function will be executed by one of the worker threads.
 * Note: This is fire-and-forget - we don't wait for completion here.
 */
int thread_pool_submit(ThreadPool* pool, void* (*fn)(void*), void* arg);

/**
 * @brief Wait for all submitted tasks to complete
 * @param pool Thread pool
 * 
 * Blocks until all tasks have been executed.
 */
void thread_pool_wait(ThreadPool* pool);

/**
 * @brief Destroy the thread pool
 * @param pool Thread pool to destroy
 * 
 * Waits for pending work, then terminates all worker threads.
 */
void thread_pool_destroy(ThreadPool* pool);

#endif /* THREAD_POOL_H */