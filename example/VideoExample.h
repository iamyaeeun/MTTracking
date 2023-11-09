#pragma once

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <atomic>

#include "BaseDetector.h"
#include "Ctracker.h"
#include "FileLogger.h"


///
/// \brief The Frame struct
///
class Frame
{
public:
    Frame() = default;
    Frame(cv::Mat imgBGR)
    {
        m_mBGR = imgBGR;
    }

    ///
    bool empty() const
    {
        return m_mBGR.empty();
    }

    ///
    const cv::Mat& GetMatBGR()
    {
        return m_mBGR;
    }

    ///
    cv::Mat& GetMatBGRWrite()
    {
        m_umBGRGenerated = false;
        m_mGrayGenerated = false;
        m_umGrayGenerated = false;
        return m_mBGR;
    }
    ///
    const cv::Mat& GetMatGray()
    {
        if (m_mGray.empty() || !m_mGrayGenerated)
        {
            if (m_umGray.empty() || !m_umGrayGenerated)
                cv::cvtColor(m_mBGR, m_mGray, cv::COLOR_BGR2GRAY);
            else
                m_mGray = m_umGray.getMat(cv::ACCESS_READ);
            m_mGrayGenerated = true;
        }
        return m_mGray;
    }
    ///
    const cv::UMat& GetUMatBGR()
    {
        std::thread::id lastThreadID = std::this_thread::get_id();

#ifdef PERF_INTER_THREAD_UMAT
        if (m_umBGR.empty() || !m_umBGRGenerated)
#else
        if (m_umBGR.empty() || !m_umBGRGenerated || lastThreadID != m_umBGRThreadID)
#endif
        {
            m_umBGR = m_mBGR.getUMat(cv::ACCESS_READ);
            m_umBGRGenerated = true;
            m_umBGRThreadID = lastThreadID;
        }
        return m_umBGR;
    }
    ///
    const cv::UMat& GetUMatGray()
    {
        std::thread::id lastThreadID = std::this_thread::get_id();

        if (m_umGray.empty() || !m_umGrayGenerated || lastThreadID != m_umGrayThreadID)
        {
            if (m_mGray.empty() || !m_mGrayGenerated)
            {
                if (m_umBGR.empty() || !m_umBGRGenerated || lastThreadID != m_umGrayThreadID)
                    cv::cvtColor(m_mBGR, m_umGray, cv::COLOR_BGR2GRAY);
                else
                    cv::cvtColor(m_umBGR, m_umGray, cv::COLOR_BGR2GRAY);
            }
            else
            {
                m_umGray = m_mGray.getUMat(cv::ACCESS_READ);
            }
            m_umGrayGenerated = true;
            m_umGrayThreadID = lastThreadID;
        }
        return m_umGray;
    }

    ///
    void SetMatBGR(const cv::Mat& imgBGR)
    {
        if (!empty())
        {
            m_mBGR.release();
        }

        m_mBGR = imgBGR;
    }

    ///
    void ReleaseMatBGR(void)
    {
        if (!empty())
        {
            m_mBGR.release();
        }
    }

    ///
    void ReleaseUMatBGR(void)
    {
        if (!m_umBGR.empty())
        {
            m_umBGR.release();
        }
    }

private:
    cv::Mat m_mBGR;
    cv::Mat m_mGray;
    cv::UMat m_umBGR;
    cv::UMat m_umGray;
    bool m_umBGRGenerated = false;
    bool m_mGrayGenerated = false;
    bool m_umGrayGenerated = false;
    std::thread::id m_umBGRThreadID;
    std::thread::id m_umGrayThreadID;
};

///
/// \brief The FrameInfo struct
///
struct FrameInfo
{
    ///
    FrameInfo()
    {
        m_frames.reserve(m_batchSize);
        m_regions.reserve(m_batchSize);
        m_frameInds.reserve(m_batchSize);
    }
    ///
    FrameInfo(size_t batchSize)
        : m_batchSize(batchSize)
    {
        m_frames.reserve(m_batchSize);
        m_regions.reserve(m_batchSize);
        m_frameInds.reserve(m_batchSize);
    }

    ///
    void SetBatchSize(size_t batchSize)
    {
        m_batchSize = batchSize;
        m_frames.reserve(m_batchSize);
        m_regions.reserve(m_batchSize);
        m_frameInds.reserve(m_batchSize);
    }

    ///
    void CleanRegions()
    {
        if (m_regions.size() != m_batchSize)
            m_regions.resize(m_batchSize);
        for (auto& regions : m_regions)
        {
            regions.clear();
        }
    }

    ///
    void CleanTracks()
    {
        if (m_tracks.size() != m_batchSize)
            m_tracks.resize(m_batchSize);
        for (auto& tracks : m_tracks)
        {
            tracks.clear();
        }
    }

    std::vector<Frame> m_frames;
    std::vector<regions_t> m_regions;
    std::vector<std::vector<TrackingObject>> m_tracks;
    std::vector<int> m_frameInds;

    size_t m_batchSize = 1;

    int64 m_dt = 0;

    std::condition_variable m_cond;
    std::mutex m_mutex;
    std::atomic<bool> m_captured { false };
};

#ifdef FULL_PIPELINE_PROCESS
struct FrameEntry
{
    ///
    FrameEntry()
    {
        m_frames.reserve(m_batchSize);
        m_regions.reserve(m_batchSize);
        m_frameInds.reserve(m_batchSize);
    }

    ///
    //FrameEntry(size_t frameInd)
    FrameEntry(size_t batchSize)
    {
        m_batchSize = MAX(1, batchSize);
        m_frames.reserve(m_batchSize);
        m_regions.reserve(m_batchSize);
        m_frameInds.reserve(m_batchSize);
    }

    ///
    void SetFrame(Frame& frame)
    {
        m_frames.push_back(frame);
        m_frameInds.push_back(0);
    }

    void SetFrame(Frame& frame, int frameIndex)
    {
        m_frames.push_back(frame);
        m_frameInds.push_back(frameIndex);
    }

    void SetImage(cv::Mat& image)
    {
        m_frames.back().SetMatBGR(image);
    }

    void SetImage(int index, cv::Mat& image)
    {
        if (index < m_batchSize)
        {
            m_frames[index].SetMatBGR(image);
        }
    }

    void SetFrameIndex(int frameIndex)
    {
        m_frameInds.back() = frameIndex;
    }

    void SetFrameIndex(int index, int frameIndex)
    {
        if (index < m_batchSize)
        {
            m_frameInds[index] = frameIndex;
        }
    }

    ///
    void SetBatchSize(size_t batchSize)
    {
        m_batchSize = batchSize;
        m_frames.reserve(m_batchSize);
        m_regions.reserve(m_batchSize);
        m_frameInds.reserve(m_batchSize);
    }

    ///
    void CleanRegions()
    {
        if (m_regions.size() != m_batchSize)
            m_regions.resize(m_batchSize);
        for (auto& regions : m_regions)
        {
            regions.clear();
        }
    }

    ///
    void CleanTracks()
    {
        if (m_tracks.size() != m_batchSize)
            m_tracks.resize(m_batchSize);
        for (auto& tracks : m_tracks)
        {
            tracks.clear();
        }
    }

    std::vector<Frame> m_frames;
    std::vector<regions_t> m_regions;
    std::vector<std::vector<TrackingObject>> m_tracks;
    std::vector<int> m_frameInds;

    std::condition_variable m_cond;
    std::mutex m_mutex;

    size_t m_batchSize = 1;     // almost constant in this design
    int64 m_procTime = 0;
};


#include "PipelineQueue.h"

#endif // FULL_PIPELINE_PROCESS

///
/// \brief The VideoExample class
///
class VideoExample
{
public:
    VideoExample(const cv::CommandLineParser& parser);
    VideoExample(const VideoExample&) = delete;
    VideoExample(VideoExample&&) = delete;
    VideoExample& operator=(const VideoExample&) = delete;
    VideoExample& operator=(VideoExample&&) = delete;

    virtual ~VideoExample() = default;

    void AsyncProcess();
    void SyncProcess();
    void PipelineProcess();

protected:
    std::unique_ptr<BaseTracker> m_tracker;

#ifdef PERF_MULTI_DETECTOR
    int m_numDetector = 1;
    std::vector<std::unique_ptr<BaseDetector>> m_detectors;
#else
    std::unique_ptr<BaseDetector> m_detector;
#endif
    bool m_isRtsp = 0;
    int m_vOnnx = 0;

    bool m_showLogs = true;
    float m_fps = 25;

	size_t m_batchSize = 1;

    int m_captureTimeOut = 90000; //60000;
    int m_trackingTimeOut = 60000;

    ResultsLog m_resultsLog;

    static void CaptureAndDetect(VideoExample* thisPtr, std::atomic<bool>& stopCapture);
#ifdef FULL_PIPELINE_PROCESS
    static void PipelineCapture(VideoExample* thisPtr, std::atomic<bool>& stopCapture);
    static void PipelineDetection(VideoExample* thisPtr, std::atomic<bool>& stopCapture);
    static void PipelineTracking(VideoExample* thisPtr, std::atomic<bool>& stopCapture);
    static void PipelineControl(VideoExample* thisPtr, std::atomic<bool>& stopCapture);

    void Detection(FrameEntry& frame);
    void Tracking(FrameEntry& frame);

    // _lkh test pipeline sync control
    mutable std::mutex m_mutex_pipe_control;
    std::condition_variable m_cond_pipe_control;
#endif

    virtual bool InitDetector(cv::UMat frame) = 0;
    virtual bool InitTracker(cv::UMat frame) = 0;

    void Detection(FrameInfo& frame);
    void Tracking(FrameInfo& frame);

    virtual void DrawData(cv::Mat frame, const std::vector<TrackingObject>& tracks, int framesCounter, int currTime) = 0;

    virtual void DrawTrack(cv::Mat frame, const TrackingObject& track, bool drawTrajectory, int framesCounter);

#ifdef FULL_PIPELINE_PROCESS
    void DrawDetect(cv::Mat frame, const regions_t& region);
    void InitFrameEntryQueue(int numBuffers = PIPELINE_FRAME_ENTRY_MAX);
#endif

    TrackerSettings m_trackerSettings;
    bool m_trackerSettingsLoaded = false;

    std::vector<cv::Scalar> m_colors;

private:
	std::vector<TrackingObject> m_tracks;

    bool m_isTrackerInitialized = false;
    bool m_isDetectorInitialized = false;
    std::string m_inFile;
    std::string m_outFile;
    int m_fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    int m_startFrame = 0;
    int m_endFrame = 0;
    int m_finishDelay = 0;
    FrameInfo m_frameInfo[2];

    bool OpenCapture(cv::VideoCapture& capture);
    bool WriteFrame(cv::VideoWriter& writer, const cv::Mat& frame);

#ifdef FULL_PIPELINE_PROCESS
    PipelineQueue m_framesQueCapture;       // Free frame queue between Display thread and Capture thread
    PipelineQueue m_framesQueDetect;        // Queue between Capture thread and Detect thread
    PipelineQueue m_framesQueTrack;         // Queue between Detect thread and Track thread
    PipelineQueue m_framesQueDisplay;       // Queue between Track thread and Display thread
#endif
};

///
void DrawFilledRect(cv::Mat& frame, const cv::Rect& rect, cv::Scalar cl, int alpha);
