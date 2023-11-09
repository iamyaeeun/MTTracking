#include "CarsCounting.h"
#include <inih/INIReader.h>

///
/// \brief CarsCounting::CarsCounting
/// \param parser
///
CarsCounting::CarsCounting(const cv::CommandLineParser& parser)
    : VideoExample(parser)
{
#ifdef _WIN32
    std::string pathToModel = "C:/Multitarget-tracker/data/";
#else
    std::string pathToModel = "../data/";
#endif

    m_drawHeatMap = parser.get<int>("heat_map") != 0;

    m_weightsFile = parser.get<std::string>("weights");
    m_configFile = parser.get<std::string>("config");
    m_namesFile = parser.get<std::string>("names");
    if (m_weightsFile.empty() && m_configFile.empty())
    {
        m_weightsFile = pathToModel + "yolov4.weights";
        m_configFile = pathToModel + "yolov4.cfg";
    }
    if (m_namesFile.empty())
        m_namesFile = pathToModel + "coco.names";

    std::map<std::string, tracking::Detectors> infMap;
    infMap.emplace("darknet", tracking::Detectors::Yolo_Darknet);
    infMap.emplace("tensorrt", tracking::Detectors::Yolo_TensorRT);
    infMap.emplace("ocvdnn", tracking::Detectors::DNN_OCV);
    std::string inference = parser.get<std::string>("inference");
    auto infType = infMap.find(inference);
    if (infType != std::end(infMap))
        m_detectorType = infType->second;
    else
        m_detectorType = tracking::Detectors::Yolo_Darknet;

    std::cout << "Inference framework set " << inference << " used " << m_detectorType << ", weights: " << m_weightsFile << ", config: " << m_configFile << ", names: " << m_namesFile << std::endl;

    m_geoBindFile = parser.get<std::string>("geo_bind");

    m_batchSize = 1;

#ifdef PERF_MULTI_DETECTOR
    // can create multiple detector
    m_numDetector = 1;
#endif
}

///
/// \brief CarsCounting::DrawTrack
/// \param frame
/// \param track
/// \param drawTrajectory
/// \param framesCounters
///
void CarsCounting::DrawTrack(cv::Mat frame, const TrackingObject& track, bool drawTrajectory, int framesCounter)
{
    cv::Rect brect = track.m_rrect.boundingRect();

    m_resultsLog.AddTrack(framesCounter, track.m_ID, brect, track.m_type, track.m_confidence);
    m_resultsLog.AddRobustTrack(track.m_ID);

    if (track.m_isStatic)
    {
#if (CV_VERSION_MAJOR >= 4)
        cv::rectangle(frame, brect, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
#else
        cv::rectangle(frame, brect, cv::Scalar(255, 0, 255), 2, CV_AA);
#endif
    }
    else
    {
#if (CV_VERSION_MAJOR >= 4)
        cv::rectangle(frame, brect, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
#else
        cv::rectangle(frame, brect, cv::Scalar(0, 255, 0), 1, CV_AA);
#endif

        if (!m_geoParams.Empty())
        {
            int traceSize = static_cast<int>(track.m_trace.size());
            int period = std::min(2 * cvRound(m_fps), traceSize);
            const auto& from = m_geoParams.Pix2Geo(track.m_trace[traceSize - period]);
            const auto& to = m_geoParams.Pix2Geo(track.m_trace[traceSize - 1]);
            auto dist = DistanceInMeters(from, to);

            std::stringstream label;
            if (period >= cvRound(m_fps) / 4)
            {
                auto velocity = (3.6f * dist * m_fps) / period;
                //std::cout << TypeConverter::Type2Str(track.m_type) << ": distance " << std::fixed << std::setw(2) << std::setprecision(2) << dist << " on time " << (period / m_fps) << " with velocity " << velocity << " km/h: " << track.m_confidence << std::endl;
                if (velocity < 1.f || std::isnan(velocity))
                    velocity = 0;
                //label << TypeConverter::Type2Str(track.m_type) << " " << std::fixed << std::setw(2) << std::setprecision(2) << velocity << " km/h";
                label << track.m_ID.ID2Str() << " " << TypeConverter::Type2Str(track.m_type) << " " << cvRound(velocity) << "kph";

                int baseLine = 0;
                double fontScale = 0.4;
                cv::Size labelSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_TRIPLEX, fontScale, 1, &baseLine);

                if (brect.x < 0)
                {
                    brect.width = std::min(brect.width, frame.cols - 1);
                    brect.x = 0;
                }
                else if (brect.x + brect.width >= frame.cols)
                {
                    brect.x = std::max(0, frame.cols - brect.width - 1);
                    brect.width = std::min(brect.width, frame.cols - 1);
                }
                if (brect.y - labelSize.height < 0)
                {
                    brect.height = std::min(brect.height, frame.rows - 1);
                    brect.y = labelSize.height;
                }
                else if (brect.y + brect.height >= frame.rows)
                {
                    brect.y = std::max(0, frame.rows - brect.height - 1);
                    brect.height = std::min(brect.height, frame.rows - 1);
                }
                cv::rectangle(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(200, 200, 200), cv::FILLED);
                cv::putText(frame, label.str(), brect.tl(), cv::FONT_HERSHEY_TRIPLEX, fontScale, cv::Scalar(0, 0, 0));

                if (velocity > 3)
                    AddToHeatMap(brect);
            }
        }
    }

    if (drawTrajectory)
    {
        cv::Scalar cl = m_colors[track.m_ID.ID2Module(m_colors.size())];

        for (size_t j = 0; j < track.m_trace.size() - 1; ++j)
        {
            const TrajectoryPoint& pt1 = track.m_trace.at(j);
            const TrajectoryPoint& pt2 = track.m_trace.at(j + 1);
#if (CV_VERSION_MAJOR >= 4)
            cv::line(frame, pt1.m_prediction, pt2.m_prediction, cl, 1, cv::LINE_AA);
#else
            cv::line(frame, pt1.m_prediction, pt2.m_prediction, cl, 1, CV_AA);
#endif
            if (!pt2.m_hasRaw)
            {
#if (CV_VERSION_MAJOR >= 4)
                cv::circle(frame, pt2.m_prediction, 4, cl, 1, cv::LINE_AA);
#else
                cv::circle(frame, pt2.m_prediction, 4, cl, 1, CV_AA);
#endif
            }
        }
    }
}

///
/// \brief CarsCounting::InitDetector
/// \param frame
///
bool CarsCounting::InitDetector(cv::UMat frame)
{
    config_t config;

#if 1
    switch (m_detectorType)
    {
    case tracking::Detectors::Yolo_Darknet:
        break;

    case tracking::Detectors::DNN_OCV:
#if 1
        config.emplace("dnnTarget", "DNN_TARGET_CPU");
        config.emplace("dnnBackend", "DNN_BACKEND_OPENCV");
#else
        config.emplace("dnnTarget", "DNN_TARGET_CUDA");
        config.emplace("dnnBackend", "DNN_BACKEND_CUDA");
#endif
        break;

    default:
        break;
    }

    config.emplace("modelConfiguration", m_configFile);
    config.emplace("modelBinary", m_weightsFile);
    config.emplace("classNames", m_namesFile);
    config.emplace("nmsThreshold", "0.4");
    config.emplace("swapRB", "0");
    config.emplace("maxCropRatio", "-1");
    if (m_batchSize > 1)
        config.emplace("maxBatch", std::to_string(m_batchSize));

    config.emplace("white_list", "person");
    config.emplace("white_list", "car");
    config.emplace("white_list", "bicycle");
    config.emplace("white_list", "motorbike");
    config.emplace("white_list", "bus");
    config.emplace("white_list", "truck");
    config.emplace("white_list", "vehicle");

    switch (m_vOnnx)
    {
    case 6: // yolov6 ONNX
        config.emplace("confidenceThreshold", "0.45");
        config.emplace("net_type", "YOLOV6");
        break;
    case 7: // yolov7 ONNX
        config.emplace("confidenceThreshold", "0.45");
        config.emplace("net_type", "YOLOV7");
        break;
    case 0: // Not ONNX
    default:
        config.emplace("confidenceThreshold", "0.5");
        break;
    }

#ifdef PERF_MULTI_DETECTOR
    for (int i = 0; i < m_numDetector; i++)
    {
        m_detectors.push_back(BaseDetector::CreateDetector(m_detectorType, config, frame));
    }
#else
    m_detector = BaseDetector::CreateDetector(m_detectorType, config, frame);
#endif

#else // Background subtraction

#if 1
    config.emplace("history", std::to_string(cvRound(10 * minStaticTime * m_fps)));
    config.emplace("varThreshold", "16");
    config.emplace("detectShadows", "1");
    m_detector = CreateDetector(tracking::Detectors::Motion_MOG2, config, frame);
#else
    config.emplace("minPixelStability", "15");
    config.emplace("maxPixelStability", "900");
    config.emplace("useHistory", "1");
    config.emplace("isParallel", "1");
    m_detector = CreateDetector(tracking::Detectors::Motion_CNT, config, m_useLocalTracking, frame);
#endif

#endif

#ifdef PERF_MULTI_DETECTOR
    return true;
#else
    return m_detector.operator bool();
#endif
}

///
/// \brief CarsCounting::InitTracker
/// \param grayFrame
///
bool CarsCounting::InitTracker(cv::UMat frame)
{
    if (m_drawHeatMap)
    {
        if (frame.channels() == 3)
            m_keyFrame = frame.getMat(cv::ACCESS_READ).clone();
        else
            cv::cvtColor(frame, m_keyFrame, cv::COLOR_GRAY2BGR);
        m_heatMap = cv::Mat(m_keyFrame.size(), CV_32FC1, cv::Scalar::all(0));
    }

    const int minStaticTime = 5;

    TrackerSettings settings;
    settings.SetDistance(tracking::DistJaccard);
    //settings.SetDistance(tracking::DistFeatureCos); // _lkh test to enable DistFeatureCos
    settings.m_kalmanType = tracking::KalmanLinear;
    settings.m_filterGoal = tracking::FilterCenter; // _lkh test, use FilterRect to enable KCF filter
    settings.m_lostTrackType = tracking::TrackCSRT; // Use KCF tracker for collisions resolving. Used if m_filterGoal == tracking::FilterRect
    settings.m_matchType = tracking::MatchHungrian;
    settings.m_dt = 0.3f;                           // Delta time for Kalman filter
    settings.m_accelNoiseMag = 0.2f;                // Accel noise magnitude for Kalman filter
    settings.m_distThres = 0.95f;                   // Distance threshold between region and object on two frames, _lkh test the original value is 0,7f
    settings.m_minAreaRadiusPix = frame.rows / 20.f;
    settings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames

    settings.AddNearTypes(TypeConverter::Str2Type("car"), TypeConverter::Str2Type("bus"), false);
    settings.AddNearTypes(TypeConverter::Str2Type("car"), TypeConverter::Str2Type("truck"), false);
    // _lkh improvement
    // cross near type among car, bus and truck
    settings.AddNearTypes(TypeConverter::Str2Type("bus"), TypeConverter::Str2Type("car"), false);
    settings.AddNearTypes(TypeConverter::Str2Type("bus"), TypeConverter::Str2Type("truck"), false);
    settings.AddNearTypes(TypeConverter::Str2Type("truck"), TypeConverter::Str2Type("car"), false);
    settings.AddNearTypes(TypeConverter::Str2Type("truck"), TypeConverter::Str2Type("bus"), false);
    settings.AddNearTypes(TypeConverter::Str2Type("person"), TypeConverter::Str2Type("bicycle"), true);
    settings.AddNearTypes(TypeConverter::Str2Type("person"), TypeConverter::Str2Type("motorbike"), true);

    settings.m_useAbandonedDetection = false;
    if (settings.m_useAbandonedDetection)
    {
        settings.m_minStaticTime = minStaticTime;
        settings.m_maxStaticTime = 60;
        settings.m_maximumAllowedSkippedFrames = cvRound(settings.m_minStaticTime * m_fps); // Maximum allowed skipped frames
        settings.m_maxTraceLength = 2 * settings.m_maximumAllowedSkippedFrames;        // Maximum trace length
    }
    else
    {
// _lkh test
#if 1
        const float sec = 0.6;        // max 1 seconds. original design is to set to 10
        settings.m_maximumAllowedSkippedFrames = cvRound(sec * m_fps); // Maximum allowed skipped frames
        settings.m_maxTraceLength = cvRound(m_fps / 2);              // Maximum trace length
#else
        settings.m_maximumAllowedSkippedFrames = cvRound(10 * m_fps); // Maximum allowed skipped frames
        settings.m_maxTraceLength = cvRound(4 * m_fps);              // Maximum trace length
#endif
    }

    m_tracker = BaseTracker::CreateTracker(settings);

    ReadGeobindings(frame.size());
    return true;
}

///
/// \brief CarsCounting::DrawData
/// \param frame
///
void CarsCounting::DrawData(cv::Mat frame, const std::vector<TrackingObject>& tracks, int framesCounter, int currTime)
{
    if (m_showLogs)
        std::cout << "Frame " << framesCounter << ": tracks = " << tracks.size() << ", time = " << currTime << std::endl;

#if 0 // Debug output
    if (!m_geoParams.Empty())
    {
        std::vector<cv::Point> points = m_geoParams.GetFramePoints();
        for (size_t i = 0; i < points.size(); ++i)
        {
            cv::line(frame, points[i % points.size()], points[(i + 1) % points.size()], cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    }
#endif

#ifdef _DEBUG
    std::stringstream label;
    label << "Frame: " << framesCounter;
    cv::putText(frame, label.str(), cv::Point2i(10, 10), cv::FONT_HERSHEY_TRIPLEX, 0.4f, cv::Scalar(10, 10, 10));
#endif

    for (const auto& track : tracks)
    {
        if (track.m_isStatic)
        {
            DrawTrack(frame, track, true, framesCounter);
        }
        else
        {
// _lkh debug
#if 1
            if (track.IsRobust(cvRound(m_fps / 15),          // Minimal trajectory size
                               0.6f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                               cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
#endif
            {
                DrawTrack(frame, track, true, framesCounter);

                CheckLinesIntersection(track, static_cast<float>(frame.cols), static_cast<float>(frame.rows));
            }
        }
    }
    //m_detector->CalcMotionMap(frame);

    if (!m_geoParams.Empty())
    {
        cv::Mat geoMap = m_geoParams.DrawTracksOnMap(tracks);
		if (!geoMap.empty())
		{
#ifndef SILENT_WORK
			cv::namedWindow("Geo map", cv::WINDOW_NORMAL);
			cv::imshow("Geo map", geoMap);
#endif
		}
    }

    for (const auto& rl : m_lines)
    {
        rl.Draw(frame);
    }

    DrawCount(frame);

    cv::Mat heatMap = DrawHeatMap();
#ifndef SILENT_WORK
    if (!heatMap.empty())
        cv::imshow("Heat map", heatMap);
#endif
}

void CarsCounting::DrawCount(cv::Mat frame)
{
    int baseLine = 0;
    double fontScale = 0.5;
    int thickness = 1;
    cv::Size labelSize = cv::getTextSize("SAMPLE", cv::FONT_HERSHEY_SIMPLEX, fontScale, 1, &baseLine);
 
    int cntEnter, cntExit, cntStay;
    
    for (int i = 0; i < m_lines.size(); i+=2)
    {
        if (m_lines[i].m_pt1.y < m_lines[i + 1].m_pt1.y)
        {
            cntEnter = m_lines[i].m_intersect2;
            cntExit = m_lines[i + 1].m_intersect2;
        }
        else
        {
            cntEnter = m_lines[i].m_intersect1;
            cntExit = m_lines[i + 1].m_intersect1;
        }
        cntStay = cntEnter - cntExit;
        std::string label = "Lane " + std::to_string(i / 2) + ": " + std::to_string(cntStay);
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
        cv::Point pt(600, 100 + labelSize.height * i / 2);
        cv::putText(frame, label, pt, cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(50, 50, 50), 1);
    }
}

///
/// \brief CarsCounting::AddLine
/// \param newLine
///
void CarsCounting::AddLine(const RoadLine& newLine)
{
    m_lines.push_back(newLine);
}

///
/// \brief CarsCounting::GetLine
/// \param lineUid
/// \return
///
bool CarsCounting::GetLine(unsigned int lineUid, RoadLine& line)
{
    for (const auto& rl : m_lines)
    {
        if (rl.m_uid == lineUid)
        {
            line = rl;
            return true;
        }
    }
    return false;
}

///
/// \brief CarsCounting::RemoveLine
/// \param lineUid
/// \return
///
bool CarsCounting::RemoveLine(unsigned int lineUid)
{
    for (auto it = std::begin(m_lines); it != std::end(m_lines);)
    {
        if (it->m_uid == lineUid)
            it = m_lines.erase(it);
        else
            ++it;
    }
    return false;
}

///
/// \brief CarsCounting::CheckLinesIntersection
/// \param track
///
void CarsCounting::CheckLinesIntersection(const TrackingObject& track, float xMax, float yMax)
{
    auto Pti2f = [&](cv::Point pt)
    {
        return cv::Point2f(pt.x / xMax, pt.y / yMax);
    };

    constexpr size_t minTrack = 5;
    if (track.m_trace.size() >= minTrack)
    {
        for (auto& rl : m_lines)
        {
            rl.IsIntersect(track.m_ID, Pti2f(track.m_trace[track.m_trace.size() - minTrack]), Pti2f(track.m_trace[track.m_trace.size() - 1]));
        }
    }
}

///
/// \brief CarsCounting::DrawHeatMap
///
cv::Mat CarsCounting::DrawHeatMap()
{
    cv::Mat res;
    if (!m_heatMap.empty())
    {
        cv::normalize(m_heatMap, m_normHeatMap, 255, 0, cv::NORM_MINMAX, CV_8UC1);
        cv::applyColorMap(m_normHeatMap, m_colorMap, cv::COLORMAP_HOT);
        cv::bitwise_or(m_keyFrame, m_colorMap, res);
    }
    return res;
}

///
/// \brief CarsCounting::AddToHeatMap
///
void CarsCounting::AddToHeatMap(const cv::Rect& rect)
{
    if (m_heatMap.empty())
        return;

    constexpr float w = 0.001f;
    for (int y = 0; y < rect.height; ++y)
    {
        float* heatPtr = m_heatMap.ptr<float>(rect.y + y) + rect.x;
        for (int x = 0; x < rect.width; ++x)
        {
            heatPtr[x] += w;
        }
    }
}

///
/// \brief CarsCounting::ReadGeobindings
///
bool CarsCounting::ReadGeobindings(cv::Size frameSize)
{
    bool res = true;
    INIReader reader(m_geoBindFile);
    
    int parseError = reader.ParseError();
    if (parseError < 0)
    {
        std::cerr << "GeoBindFile file " << m_geoBindFile << " does not exist!" << std::endl;
        res = false;
    }
    else if (parseError > 0)
    {
        std::cerr << "GeoBindFile file " << m_geoBindFile << " parse error in line: " << parseError << std::endl;
        res = false;
    }
    if (!res)
        return res;

    // Read frame-map bindings
    std::vector<cv::Point2d> geoPoints;
    std::vector<cv::Point> framePoints;
    for (size_t i = 0;; ++i)
    {
        cv::Point2d geoPoint;
        std::string lat = "lat" + std::to_string(i);
        std::string lon = "lon" + std::to_string(i);
        std::string px_x = "px_x" + std::to_string(i);
        std::string px_y = "px_y" + std::to_string(i);
        if (reader.HasValue("points", lat) && reader.HasValue("points", lon) && reader.HasValue("points", px_x) && reader.HasValue("points", px_y))
        {
            geoPoints.emplace_back(reader.GetReal("points", lat, 0), reader.GetReal("points", lon, 0));
            framePoints.emplace_back(cvRound(reader.GetReal("points", px_x, 0) * frameSize.width), cvRound(reader.GetReal("points", px_y, 0) * frameSize.height));
        }
        else
        {
            break;
        }
    }
    res = m_geoParams.SetKeyPoints(framePoints, geoPoints);

    // Read map image
    std::string mapFile = reader.GetString("map", "file", "");
    std::vector<cv::Point2d> mapGeoCorners;
    mapGeoCorners.emplace_back(reader.GetReal("map", "left_top_lat", 0), reader.GetReal("map", "left_top_lon", 0));
    mapGeoCorners.emplace_back(reader.GetReal("map", "right_top_lat", 0), reader.GetReal("map", "right_top_lon", 0));
    mapGeoCorners.emplace_back(reader.GetReal("map", "right_bottom_lat", 0), reader.GetReal("map", "right_bottom_lon", 0));
    mapGeoCorners.emplace_back(reader.GetReal("map", "left_bottom_lat", 0), reader.GetReal("map", "left_bottom_lon", 0));
    m_geoParams.SetMapParams(mapFile, mapGeoCorners);

    // Read lines
    std::cout <<"Read lines:" << std::endl;
    for (size_t i = 0;; ++i)
    {
        std::string line = "line" + std::to_string(i);
        std::string x0 = line + "_x0";
        std::string y0 = line + "_y0";
        std::string x1 = line + "_x1";
        std::string y1 = line + "_y1";
        if (reader.HasValue("lines", x0) && reader.HasValue("lines", y0) && reader.HasValue("lines", x1) && reader.HasValue("lines", y1))
        {
            cv::Point2f p0(reader.GetReal("lines", x0, 0), reader.GetReal("lines", y0, 0));
            cv::Point2f p1(reader.GetReal("lines", x1, 0), reader.GetReal("lines", y1, 0));
            std::cout << "Line" << i << ": " << p0 << " - " << p1 << std::endl;
            AddLine(RoadLine(p0, p1, i));
        }
        else
        {
            break;
        }
    }

    return res;
}
