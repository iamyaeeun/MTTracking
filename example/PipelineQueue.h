#pragma once

#include <queue>
#include <deque>
#include <list>
#include <vector>
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <atomic>

#define SHOW_QUE_LOG 0
#if SHOW_QUE_LOG
///
/// \brief currTime
/// \return
///
inline std::string CurrTime_()
{
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d.%m.%Y %H:%M:%S:");
    return oss.str();
}

#define QUE_LOG std::cout << CurrTime_()
#define QUE_ERR_LOG (std::cerr << CurrTime_())
#endif

#define PIPELINE_FRAME_ENTRY_MAX       32

struct FrameEntry;
typedef std::shared_ptr<FrameEntry> frame_entry_ptr;

///
/// A threadsafe-queue
///
class PipelineQueue
{
private:
    typedef std::list<frame_entry_ptr> queue_t;

public:
    ///
    /// \brief FramesQueue
    ///
    PipelineQueue()
        : m_que(),
        m_mutex(),
        m_cond_producer(),
        m_cond_consumer()
    {}

    PipelineQueue(const PipelineQueue&) = delete;
    PipelineQueue(PipelineQueue&&) = delete;
    PipelineQueue& operator=(const PipelineQueue&) = delete;
    PipelineQueue& operator=(PipelineQueue&&) = delete;

    ///
    ~PipelineQueue(void) = default;

    void PushFrameEntry(frame_entry_ptr frameEntry)
    {
        std::unique_lock<std::mutex> lock(m_mutex);

        if (m_que.size() == m_max_queue_size)
        {
            //std::cout << "producer wait" << std::endl;
            // if there no more space to push, lock producer
            m_cond_producer.wait(lock);
        }

        m_que.push_back(frameEntry);

        if (m_que.size() == 1) // first entry
        {
            //std::cout << "notifier consumer" << std::endl;
            // If the first entry is pushed, trigger consumer
            m_cond_consumer.notify_one();
        }
    }

    frame_entry_ptr PopFrameEntry(void)
    {
        frame_entry_ptr entry = nullptr;
        std::unique_lock<std::mutex> lock(m_mutex);

        if (m_que.size() == 0)
        {
            //std::cout << "consumer wait" << std::endl;
            // if the queue is empty, lock consumer till an entry is available
            m_cond_consumer.wait(lock);
        }
        
        entry = m_que.front();
        m_que.pop_front();

        if (m_que.size() == m_max_queue_size - 1)
        {
            //std::cout << "notify producer" << std::endl;
            // if there is space, notify producer to unlock
            m_cond_producer.notify_one();
        }

        return entry;
    }

private:
    queue_t m_que;
    mutable std::mutex m_mutex;

    size_t m_max_queue_size = PIPELINE_FRAME_ENTRY_MAX;

    // sync conditions for producer and consumer of the pipeline stages
    std::condition_variable m_cond_producer;
    std::condition_variable m_cond_consumer;
};
