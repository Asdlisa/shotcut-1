/*
 * beat_detection_filter.cpp
 * 
 * 这是一个 MLT 滤镜，用于检测音频中的节拍。
 * 基于 FFmpeg 的 RDFT (Real Discrete Fourier Transform) 实现。
 * 
 * 编译命令示例 (需链接 mlt++, avcodec, avutil):
 * g++ -shared -fPIC beat_detection_filter.cpp -o libmltbeat_detection.so \
 *    `mlt-config --cflags --libs` -lavcodec -lavutil -std=c++11
 */

#include <framework/mlt.h>
#include <framework/mlt_filter.h>
#include <framework/mlt_frame.h>

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

// 引入 FFmpeg 的 FFT 头文件
extern "C" {
#include <libavcodec/avfft.h>
#include <libavutil/mem.h>
}

// === 常量定义 ===
constexpr int FRAME_SIZE = 1024;  // FFT 窗口大小
constexpr int HOP_SIZE = 512;     // 滑动窗口步长
constexpr int ENERGY_HISTORY = 10;// 用于计算阈值的能量历史长度

// === 滤镜私有数据结构 ===
// 用于在滤镜实例中存储状态
struct filter_data {
    RDFTContext *rdft_ctx;         // FFmpeg RDFT 上下文
    std::vector<double> window;   // 环形缓冲区
    int write_pos;                // 写入位置
    double *fft_in;               // FFT 输入缓冲区 (对齐内存)
    double *fft_out;              // FFT 输出缓冲区
    double *magnitude;            // 幅度谱
    double *energy_history;       // 能量历史记录
    int history_count;            // 当前历史记录计数
    
    filter_data() : rdft_ctx(nullptr), write_pos(0), history_count(0) {
        window.resize(FRAME_SIZE * 2);
        // 使用 av_malloc 确保内存对齐 (FFTW/FFmpeg SSE优化需要)
        fft_in = (double*)av_malloc(sizeof(double) * FRAME_SIZE);
        fft_out = (double*)av_malloc(sizeof(double) * FRAME_SIZE);
        magnitude = (double*)av_malloc(sizeof(double) * (FRAME_SIZE / 2));
        energy_history = (double*)av_malloc(sizeof(double) * ENERGY_HISTORY);
        
        std::fill(window.begin(), window.end(), 0.0);
        std::fill(energy_history, energy_history + ENERGY_HISTORY, 0.0);
    }

    ~filter_data() {
        if (rdft_ctx) av_rdft_end(rdft_ctx);
        av_free(fft_in);
        av_free(fft_out);
        av_free(magnitude);
        av_free(energy_history);
    }
};

// === 辅助函数：将交错的音频混合为单声道 ===
void mix_to_mono(const int16_t *input, int samples, int channels, std::vector<double> &output) {
    output.resize(samples);
    for (int i = 0; i < samples; ++i) {
        double sum = 0;
        for (int ch = 0; ch < channels; ++ch) {
            sum += input[i * channels + ch];
        }
        output[i] = sum / channels;
    }
}

// === 核心：处理一帧音频 ===
static int filter_get_audio(mlt_frame frame, void **buffer, mlt_audio_format *format, int *frequency, int *channels, int *samples) {
    // 1. 获取滤镜实例
    mlt_filter filter = (mlt_filter)mlt_frame_pop_audio(frame);
    
    // 2. 获取私有数据
    filter_data *data = (filter_data*)filter->child;

    // 3. 获取原始音频数据
    // 我们强制请求 mlt_audio_s16 格式，这是最通用的整数格式
    *format = mlt_audio_s16;
    int error = mlt_frame_get_audio(frame, buffer, format, frequency, channels, samples);
    if (error) return error;

    const int16_t *audio_in = (const int16_t *)*buffer;

    // 4. 转换为浮点单声道数据
    std::vector<double> mono_samples;
    mix_to_mono(audio_in, *samples, *channels, mono_samples);

    // 5. 初始化 RDFT (如果尚未初始化)
    if (!data->rdft_ctx) {
        // DFT_R2C: Real to Complex (实数输入，复数输出)
        data->rdft_ctx = av_rdft_init(log2(FRAME_SIZE), DFT_R2C);
    }

    // 6. 将样本填入环形缓冲区
    for (int i = 0; i < *samples; ++i) {
        data->window[data->write_pos] = mono_samples[i];
        data->window[data->write_pos + FRAME_SIZE] = mono_samples[i]; // 镜像，方便读取
        data->write_pos = (data->write_pos + 1) % FRAME_SIZE;
    }

    // 7. 对每一帧的步长进行处理
    // 我们不希望对每个样本都做 FFT，太慢了。每当缓冲区填满 HOP_SIZE 时，我们处理一次。
    // 为了简化，我们这里假设 samples 比较大，或者我们只要处理最近的窗口。
    // 注意：真实的流式处理需要更复杂的计数器逻辑，这里为了演示完整性，
    // 我们只对当前缓冲区中最新的一个完整窗口进行分析。
    
    // 构造当前帧的 FFT 输入
    // 从 write_pos 回溯 FRAME_SIZE 个样本
    for (int i = 0; i < FRAME_SIZE; ++i) {
        int read_pos = (data->write_pos - FRAME_SIZE + i + FRAME_SIZE) % FRAME_SIZE;
        data->fft_in[i] = data->window[read_pos];
    }

    // 8. 执行 FFT
    if (data->rdft_ctx) {
        av_rdft_calc(data->rdft_ctx, data->fft_in, data->fft_out);

        // 9. 计算幅度谱
        // FFmpeg RDFT 输出: [r0, re1, im1, re2, im2, ...]
        for (int i = 0; i < FRAME_SIZE / 2; ++i) {
            double re = data->fft_out[2 * i];
            double im = data->fft_out[2 * i + 1];
            data->magnitude[i] = sqrt(re * re + im * im);
        }

        // 10. 计算能量
        double current_energy = 0;
        for (int i = 0; i < FRAME_SIZE / 2; ++i) {
            current_energy += data->magnitude[i];
        }

        // 11. 节拍检测逻辑
        double threshold = 0.0;
        double sum = 0.0;
        int count = 0;
        for (int i = 0; i < ENERGY_HISTORY; ++i) {
            sum += data->energy_history[i];
            if (data->energy_history[i] > 0.0001) count++;
        }
        
        if (count > 0) {
            double avg = sum / count;
            threshold = avg * 1.5; // 简单的自适应阈值
        } else {
            threshold = 0.01;
        }

        // 12. 更新历史
        // 滚动更新
        for (int i = 0; i < ENERGY_HISTORY - 1; ++i) {
            data->energy_history[i] = data->energy_history[i+1];
        }
        data->energy_history[ENERGY_HISTORY - 1] = current_energy;

        // 13. 判断是否为节拍
        if (current_energy > threshold) {
            // 在这里触发节拍事件
            // 例如：打印日志，或者通过 mlt_properties_set 设置一个属性供 QML 检测
            // mlt_properties props = MLT_FILTER_PROPERTIES(filter);
            // mlt_properties_set_double(props, "beat_detected", mlt_properties_get_double(props, "_time"));
            
            fprintf(stderr, "BEAT DETECTED! Energy: %f, Threshold: %f\n", current_energy, threshold);
        }
    }

    return 0;
}

// === MLT 滤镜构造函数 (入口点) ===
extern "C" {
    
    // 初始化函数
    mlt_filter filter_beat_detection_init(mlt_profile profile, mlt_service_type type, const char *id, char *arg) {
        // 创建滤镜对象
        mlt_filter filter = mlt_filter_new();
        if (!filter) return NULL;

        // 分配私有数据
        filter_data *data = new filter_data();
        if (!data) {
            mlt_filter_close(filter);
            return NULL;
        }
        
        filter->child = data;
        
        // 设置处理音频的回调
        // 当 MLT 管道请求音频时，会调用这个函数
        filter->process = filter_get_audio;

        return filter;
    }

    // 关闭函数 (可选，MLT 会在 service 关闭时自动调用 child 的析构函数，但清理工作最好显式做)
    void filter_beat_detection_close(mlt_filter filter) {
        if (filter) {
            delete (filter_data*)filter->child;
            filter->child = nullptr;
            mlt_filter_close(filter);
        }
    }
}
