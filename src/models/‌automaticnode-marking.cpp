/*
 * automaticnodemarking.cpp
 *
 * 功能：基于 MLT 框架的节拍检测与分析模块
 * 说明：
 * 1. 这是一个基于能量检测的简单节拍器。
 * 2. 它使用 FFmpeg (libavcodec) 进行 FFT 变换。
 * 3. 它设计为可以被 MLT Filter 调用，或者作为一个独立的分析工具。
 */

#include "models/models.h"
#include <framework/mlt.h>
#include <framework/mlt_filter.h>
#include <framework/mlt_frame.h>
#include <framework/mlt_log.h>

#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>

// FFmpeg FFT 头文件
extern "C" {
#include <libavcodec/avfft.h>
#include <libavutil/mem.h>
}

// === 配置常量 ===
constexpr int FFT_SIZE = 1024;  // FFT 窗口大小
constexpr int HOP_SIZE = 512;   // 滑动窗口步长
constexpr int ENERGY_BUFFER_SIZE =
    43;  // 约 1 秒的历史能量 (43 * 512 / 44100 ≈ 0.5s，根据采样率调整)
constexpr double C = 1.3;  // 常数系数，用于计算自适应阈值

// === 私有数据结构 ===
typedef struct {
  RDFTContext *rdft;      // FFmpeg RDFT 上下文
  double *fft_in;         // FFT 输入缓冲区
  double *fft_out;        // FFT 输出缓冲区
  double *energy_buffer;  // 能量历史记录
  int energy_index;       // 当前能量缓冲区写入位置
  double avg_energy;      // 平均能量
  bool is_initialized;    // 初始化标志
} BeatMarkerPrivate;

// 初始化私有数据
static void private_init(BeatMarkerPrivate *self) {
  self->rdft = av_rdft_init(log2(FFT_SIZE), DFT_R2C);
  self->fft_in = (double *)av_malloc(sizeof(double) * FFT_SIZE);
  self->fft_out = (double *)av_malloc(sizeof(double) * FFT_SIZE);
  self->energy_buffer =
      (double *)av_malloc(sizeof(double) * ENERGY_BUFFER_SIZE);

  // 清零
  memset(self->energy_buffer, 0, sizeof(double) * ENERGY_BUFFER_SIZE);
  self->energy_index = 0;
  self->avg_energy = 0.0;
  self->is_initialized = (self->rdft != nullptr);
}

// 释放私有数据
static void private_close(BeatMarkerPrivate *self) {
  if (self->rdft) av_rdft_end(self->rdft);
  if (self->fft_in) av_free(self->fft_in);
  if (self->fft_out) av_free(self->fft_out);
  if (self->energy_buffer) av_free(self->energy_buffer);
  self->is_initialized = false;
}

// === 核心算法：计算频谱能量 ===
static double calculate_band_energy(const double *fft_out, int low_bin,
                                   int high_bin) {
  double energy = 0.0;
  for (int i = low_bin; i <= high_bin && i < FFT_SIZE / 2; ++i) {
    double re = fft_out[2 * i];
    double im = fft_out[2 * i + 1];
    energy += sqrt(re * re + im * im);
  }
  return energy;
}

// === 核心：检测节拍 ===
// 输入：单声道音频数据，样本数
// 返回：检测到节拍返回 true
static bool detect_beat(BeatMarkerPrivate *self, const double *samples,
                        int num_samples, int sample_rate) {
  if (!self->is_initialized) return false;

  // 1. 环形缓冲区填充 (简化版：假设 num_samples == HOP_SIZE)
  // 实际生产中需要维护一个环形缓冲区，这里为了演示，直接使用传入的样本
  // 我们假设调用方已经管理好了滑动窗口

  // 2. 准备 FFT 输入 (应用汉宁窗)
  for (int i = 0; i < FFT_SIZE; ++i) {
    // 简单的汉宁窗窗函数
    double window = 0.5 * (1.0 - cos(2.0 * M_PI * i / (FFT_SIZE - 1)));
    self->fft_in[i] = samples[i] * window;
  }

  // 3. 执行 FFT
  av_rdft_calc(self->rdft, self->fft_in, self->fft_out);

  // 4. 计算低频能量 (通常是鼓点所在的频段)
  // 假设采样率 44100，FFT_SIZE 1024，每个 bin 约 43Hz
  // Bass 通常在 0 - 250Hz 之间，即 bin 0 - 6
  double current_energy = calculate_band_energy(self->fft_out, 1, 5);

  // 5. 更新平均能量 (简单移动平均)
  self->energy_buffer[self->energy_index] = current_energy;
  self->energy_index = (self->energy_index + 1) % ENERGY_BUFFER_SIZE;

  // 计算缓冲区内的局部平均能量
  double local_avg = 0.0;
  for (int i = 0; i < ENERGY_BUFFER_SIZE; ++i) {
    local_avg += self->energy_buffer[i];
  }
  local_avg /= ENERGY_BUFFER_SIZE;

  // 6. 节拍判定逻辑
  // 如果当前能量 > 平均能量 * 常数，则认为是节拍
  // 为了避免重复触发，这里可以加入冷却时间逻辑
  bool is_beat = false;
  if (current_energy > local_avg * C) {
    // 这里可以添加：局部峰值检测
    // 比如 current_energy > prev_energy && current_energy > next_energy
    is_beat = true;
  }

  // 调试输出
  // mlt_log_info(NULL, "Energy: %f, Avg: %f, Beat: %d\n", current_energy,
  // local_avg, is_beat);

  return is_beat;
}

// === MLT 滤镜接口 (集成部分) ===

// 获取音频的回调函数
static int filter_get_audio(mlt_frame frame, void **buffer,
                           mlt_audio_format *format, int *frequency,
                           int *channels, int *samples) {
  // 1. 获取滤镜实例
  mlt_filter filter = (mlt_filter)mlt_frame_pop_audio(frame);

  // 2. 获取私有数据
  BeatMarkerPrivate *self = (BeatMarkerPrivate *)filter->child;
  if (!self) {
    // 如果还没初始化，初始化它 (注意：这里逻辑其实有缺陷，正常流程应该在 init 中做)
    // 但为了防止崩溃，做临时保护
    return 0;
  }

  // 3. 获取原始音频数据
  // 请求 s16 格式，这是最通用的
  *format = mlt_audio_s16;
  int error = mlt_frame_get_audio(frame, buffer, format, frequency, channels,
                                  samples);
  if (error) return error;

  // 4. 转换为双精度浮点并进行声道混合
  // 为了计算 FFT，我们需要连续的单声道浮点数据
  const int16_t *audio_in = (const int16_t *)*buffer;

  // 临时的单声道缓冲区
  std::vector<double> mono_samples(*samples);
  for (int i = 0; i < *samples; ++i) {
    double sum = 0;
    for (int ch = 0; ch < *channels; ++ch) {
      sum += audio_in[i * *channels + ch];
    }
    mono_samples[i] = sum / *channels;
  }

  // 5. 调用核心检测算法
  // 注意：这里每次处理的是一帧（例如 4096 个样本），而我们的 FFT 窗口只有 1024
  // 真实的实现需要在一个大的环形缓冲区中滑动处理
  // 这里为了演示，我们只取这一帧的前 1024 个样本进行处理
  int samples_to_process = std::min(*samples, FFT_SIZE);

  if (detect_beat(self, mono_samples.data(), samples_to_process, *frequency)) {
    // === 检测到节拍！ ===

    // 获取当前帧的时间位置 (秒)
    mlt_position position = mlt_frame_get_position(frame);
    double time_sec =
        (double)position / mlt_profile_fps(mlt_frame_profile(frame));

    mlt_log_info(NULL, "[BeatMarker] BEAT DETECTED at %.2f sec\n", time_sec);

    // TODO: 在这里可以将时间信息写入 filter 的属性中
    // mlt_properties props = MLT_FILTER_PROPERTIES(filter);
    // mlt_properties_set_double(props, "_last_beat_time", time_sec);
  }

  return 0;
}

// === 构造函数与析构函数 (C 风格导出) ===

extern "C" {

mlt_filter beatmarker_filter_init(mlt_profile profile, mlt_service_type type,
                                 const char *id, char *arg) {
  mlt_filter filter = mlt_filter_new();
  if (!filter) return NULL;

  // 分配并初始化私有数据
  BeatMarkerPrivate *self = new BeatMarkerPrivate();
  private_init(self);

  filter->child = self;
  filter->process = filter_get_audio;  // 注册音频处理回调

  return filter;
}

void beatmarker_filter_close(mlt_filter filter) {
  if (filter) {
    BeatMarkerPrivate *self = (BeatMarkerPrivate *)filter->child;
    private_close(self);
    delete self;
    filter->child = nullptr;
    mlt_filter_close(filter);
  }
}
}
