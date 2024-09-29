/**
 * @file fitsine.h
 * @author 黄裕炯 (961352855@qq.com)
 * @brief
 * @version 1.0
 * @date 2022-09-04
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#pragma once

#include <array>
#include <cstdint>

//! @addtogroup fitsine_filter
//! @{

/**
 * @brief The function of fourParaFitEigen \cite FitSine
 * @note \f$Acos(ωt) + Bsin(ωt) + C\f$
 *
 * @param datacount
 * @param data_arr
 * @param Fs
 * @param CWFreq
 * @param FreResolution
 * @param GridSize
 * @return std::array<float, 4>
 */
std::array<float, 4> fourParaFitEigen(std::size_t datacount,
                                      float *data_arr,
                                      const float &Fs,
                                      const float &CWFreq,
                                      const float &FreResolution,
                                      const uint32_t &GridSize);

//! @} fitsine_filter
