/**
 * @file thread_pool.c
 * @brief Thread pool implementation using pthreads
 * 
 * THREADING EXPLAINED:
 * ===================
 * 
 * A THREAD is like a worker in a factory. Each thread can run independently,
 * doing its own work while other threads do theirs.
 * 
 * Without threads (sequential):
 *   Task A → Task B → Task C  (one after another)
 * 
 * With threads (parallel):
 *   Thread 1: Task A → 
 *   Thread 2: Task B →   (all at once!)
 *   Thread 3: Task C →
 * 
 * RACE CONDITION EXAMPLE (DELIBERATELY BUGGY!):
 * =============================================
 * 
 * int counter = 0;
 * 
 * // Thread 1:
 * void* worker1(void* arg) {
 *     for (int i = 0; i < 1000000; i++) {
 *         counter = counter + 1;  // NOT ATOMIC!
 *     }
 * }
 * 
 * // Thread 2:
 * void* worker2(void* arg) {
 *     for (int i = 0; i < 1000000; i++) {
 *         counter = counter + 1;
 *     }
 * }
 * 
 * Expected: counter = 2000000
 * Actual: counter = ~1200000 (lost ~800k updates!)
 * 
 * WHY? The increment takes 3 steps:
 *   1. READ counter from memory (e.g., 100)
 *   2. ADD 1 (now 101)
 *   3. WRITE back to memory (101)
 * 
 * If both threads read 100 before either writes, both write 101!
 * 
 * FIX: Use a mutex to protect counter
 * 
 * MUTEX ANALOGY: A bathroom lock
 * ==============================
 * Only one person can enter the bathroom at a time.
 * They lock the door, do their business, then unlock.
 * Everyone else waits outside until it's their turn.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "thread_pool.h"

/**
 * Work item structure - a task in the queue
 */
typedef struct WorkItem {
    void* (*fn)(void*);    /**< Function to execute */
    void* arg;             /**< Argument to function */
    struct WorkItem* next; /**< Next item in queue */
} WorkItem;

/**
 * Thread pool structure
 */
struct ThreadPool {
    pthread_t* threads;        /**< Array of worker threads */
    int num_threads;           /**< Number of threads */
    
    /* Work queue */
    WorkItem* work_head;      /**< Head of work queue */
    WorkItem* work_tail;      /**< Tail of work queue */
    int work_count;            /**< Number of pending work items */
    
    /* Synchronization */
    pthread_mutex_t mutex;    /**< Protects work queue */
    pthread_cond_t work_cv;   /**< Signal when work available */
    pthread_cond_t done_cv;    /**< Signal when all work done */
    
    /* State */
    int shutdown;             /**< Pool is shutting down */
};

/**
 * Worker thread function
 * 
 * Each worker loops forever, waiting for work, doing work, repeating.
 */
static void* worker_thread(void* arg)
{
    ThreadPool* pool = (ThreadPool*)arg;
    
    while (1) {
        /* Lock the queue */
        pthread_mutex_lock(&pool->mutex);
        
        /* Wait for work (or shutdown) */
        while (pool->work_count == 0 && !pool->shutdown) {
            pthread_cond_wait(&pool->work_cv, &pool->mutex);
        }
        
        /* Check if we should exit */
        if (pool->shutdown && pool->work_count == 0) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }
        
        /* Get work item from queue */
        WorkItem* item = pool->work_head;
        if (item != NULL) {
            pool->work_head = item->next;
            if (pool->work_head == NULL) {
                pool->work_tail = NULL;
            }
            pool->work_count--;
        }
        
        /* Signal if queue is empty (for wait) */
        if (pool->work_count == 0) {
            pthread_cond_signal(&pool->done_cv);
        }
        
        pthread_mutex_unlock(&pool->mutex);
        
        /* Execute the work (outside the lock!) */
        if (item != NULL) {
            item->fn(item->arg);
            free(item);
        }
    }
    
    return NULL;
}

/**
 * Create a thread pool with specified number of threads
 */
ThreadPool* thread_pool_create(int num_threads)
{
    if (num_threads <= 0) {
        num_threads = 4;  /* Default to 4 threads */
    }
    
    ThreadPool* pool = (ThreadPool*)malloc(sizeof(ThreadPool));
    if (pool == NULL) {
        return NULL;
    }
    
    /* Initialize fields */
    pool->num_threads = num_threads;
    pool->work_head = NULL;
    pool->work_tail = NULL;
    pool->work_count = 0;
    pool->shutdown = 0;
    
    /* Initialize synchronization primitives */
    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->work_cv, NULL);
    pthread_cond_init(&pool->done_cv, NULL);
    
    /* Allocate thread array */
    pool->threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    if (pool->threads == NULL) {
        free(pool);
        return NULL;
    }
    
    /* Create worker threads */
    for (int i = 0; i < num_threads; i++) {
        if (pthread_create(&pool->threads[i], NULL, worker_thread, pool) != 0) {
            /* On failure, cleanup what we have */
            pool->shutdown = 1;
            pthread_cond_broadcast(&pool->work_cv);
            for (int j = 0; j < i; j++) {
                pthread_join(pool->threads[j], NULL);
            }
            free(pool->threads);
            free(pool);
            return NULL;
        }
    }
    
    return pool;
}

/**
 * Get the number of threads in the pool
 */
int thread_pool_get_num_threads(ThreadPool* pool)
{
    if (pool == NULL) {
        return 0;
    }
    return pool->num_threads;
}

/**
 * Submit a task to the thread pool
 */
int thread_pool_submit(ThreadPool* pool, void* (*fn)(void*), void* arg)
{
    if (pool == NULL || fn == NULL) {
        return -1;
    }
    
    /* Create work item */
    WorkItem* item = (WorkItem*)malloc(sizeof(WorkItem));
    if (item == NULL) {
        return -1;
    }
    
    item->fn = fn;
    item->arg = arg;
    item->next = NULL;
    
    /* Add to queue */
    pthread_mutex_lock(&pool->mutex);
    
    if (pool->work_tail != NULL) {
        pool->work_tail->next = item;
    } else {
        pool->work_head = item;
    }
    pool->work_tail = item;
    pool->work_count++;
    
    /* Signal a worker that work is available */
    pthread_cond_signal(&pool->work_cv);
    
    pthread_mutex_unlock(&pool->mutex);
    
    return 0;
}

/**
 * Wait for all submitted tasks to complete
 */
void thread_pool_wait(ThreadPool* pool)
{
    if (pool == NULL) {
        return;
    }
    
    pthread_mutex_lock(&pool->mutex);
    
    /* Wait until queue is empty */
    while (pool->work_count > 0) {
        pthread_cond_wait(&pool->done_cv, &pool->mutex);
    }
    
    pthread_mutex_unlock(&pool->mutex);
}

/**
 * Destroy the thread pool
 */
void thread_pool_destroy(ThreadPool* pool)
{
    if (pool == NULL) {
        return;
    }
    
    /* Signal shutdown */
    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->work_cv);
    pthread_mutex_unlock(&pool->mutex);
    
    /* Wait for all threads to exit */
    for (int i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    
    /* Cleanup remaining work items */
    WorkItem* item = pool->work_head;
    while (item != NULL) {
        WorkItem* next = item->next;
        free(item);
        item = next;
    }
    
    /* Destroy synchronization primitives */
    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->work_cv);
    pthread_cond_destroy(&pool->done_cv);
    
    /* Free memory */
    free(pool->threads);
    free(pool);
}