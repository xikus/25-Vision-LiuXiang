/**
 * @file dataio.cpp
 * @author zhaoxi (535394140@qq.com)
 * @brief
 * @version 1.0
 * @date 2023-04-22
 *
 * @copyright Copyright 2023 (c), zhaoxi
 *
 */

#include <opencv2/core.hpp>

#include "srvl/core/util.hpp"
#include "srvl/dataio/dataio.h"

using namespace std;
using namespace cv;

inline void writeTR(FileStorage &fs, const GyroData::Translation &t, const GyroData::Rotation &r)
{
    fs << "translation"
       << "{"
       << "x" << t.x << "y" << t.y << "z" << t.z
       << "vx" << t.vx << "vy" << t.vy << "vz" << t.vz
       << "}"
       << "rotation"
       << "{"
       << "yaw" << r.yaw << "pitch" << r.pitch << "roll" << r.roll
       << "yaw_speed" << r.yaw_speed << "pitch_speed" << r.pitch_speed << "roll_speed" << r.roll_speed
       << "}";
}

inline void readTR(FileNode &fn, GyroData::Translation &t, GyroData::Rotation &r)
{
    FileNode translation_data = fn["translation"];
    FileNode rotation_data = fn["rotation"];
    translation_data["x"] >> t.x;
    translation_data["y"] >> t.y;
    translation_data["z"] >> t.z;
    translation_data["vx"] >> t.vx;
    translation_data["vy"] >> t.vy;
    translation_data["vz"] >> t.vz;
    rotation_data["yaw"] >> r.yaw;
    rotation_data["pitch"] >> r.pitch;
    rotation_data["roll"] >> r.roll;
    rotation_data["yaw_speed"] >> r.yaw_speed;
    rotation_data["pitch_speed"] >> r.pitch_speed;
    rotation_data["roll_speed"] >> r.roll_speed;
}

bool GyroData::write(const string &path, uint32_t idx, const GyroData &data) noexcept
{
    FileStorage fs(path, FileStorage::APPEND);
    if (!fs.isOpened())
        return false;
    string data_str = "gyro_data_" + to_string(idx);
    fs << data_str << "{";
    writeTR(fs, data.translation, data.rotation);
    fs << "}";
    return true;
}

bool GyroData::read(const string &path, uint32_t idx, GyroData &data) noexcept
{
    FileStorage fs(path, FileStorage::READ);
    if (!fs.isOpened())
        return false;
    string data_str = "gyro_data_" + to_string(idx);
    FileNode data_node = fs[data_str];
    if (data_node.isNone())
        return false;
    if (data_node["translation"].isNone() || data_node["rotation"].isNone())
        return false;
    readTR(data_node, data.translation, data.rotation);
    return true;
}

bool TransferData::write(const string &path, uint32_t idx, const TransferData &data) noexcept
{
    FileStorage fs(path, FileStorage::APPEND);
    if (!fs.isOpened())
        return false;
    string data_str = "transfer_data_" + to_string(idx);
    fs << data_str << "{";
    writeTR(fs, data.translation, data.rotation);
    fs << "data_u8" << data.data_u8;
    fs << "data_f32" << data.data_f32;
    fs << "}";
    return true;
}

bool TransferData::read(const string &path, uint32_t idx, TransferData &data) noexcept
{
    FileStorage fs(path, FileStorage::READ);
    if (!fs.isOpened())
        return false;
    string data_str = "transfer_data_" + to_string(idx);
    FileNode data_node = fs[data_str];
    if (data_node.isNone())
        return false;
    readTR(data_node, data.translation, data.rotation);
    data_node["data_u8"] >> data.data_u8;
    data_node["data_f32"] >> data.data_f32;
    return true;
}

bool writeCorners(const string &path, uint32_t idx, const vector<vector<Point2d>> &corners)
{
    FileStorage fs(path, FileStorage::APPEND);
    if (!fs.isOpened())
        return false;
    fs << "corners_" + to_string(idx) << "[";
    for (size_t i = 0; i < corners.size(); ++i)
    {
        fs << "[";
        for (size_t j = 0; j < corners[i].size(); ++j)
            fs << "{"
               << "x" << corners[i][j].x << "y" << corners[i][j].y << "}";
        fs << "]";
    }
    fs << "]";
    return true;
}

bool readCorners(const string &path, uint32_t idx, vector<vector<Point2d>> &corners)
{
    FileStorage fs(path, FileStorage::READ);
    if (!fs.isOpened())
        return false;
    FileNode corners_node = fs["corners_" + to_string(idx)];
    for (auto each_node : corners_node)
    {
        if (each_node.type() != FileNode::SEQ)
            return false;
        vector<Point2d> row;
        for (auto point_node : each_node)
        {
            double x, y;
            point_node["x"] >> x;
            point_node["y"] >> y;
            row.emplace_back(x, y);
        }
        corners.emplace_back(row);
    }
    return true;
}
