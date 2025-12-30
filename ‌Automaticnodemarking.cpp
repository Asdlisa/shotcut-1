// beat_tracker.cpp
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <sndfile.h>
#include <fftw3.h>

// 常量定义
const int BUFFER_SIZE = 1024;
const int HOP_SIZE = 512;
const int FRAME_SIZE = 1024;

// 计算一个复数数组的幅度
void calculate_magnitude(fftw_complex *in, std::vector<double> &out) {
    for (size_t i = 0; i < out.size(); i++) {
        out[i] = std::sqrt(in[i][0] * in[i][0] + in[i][1] * in[i][1]);
    }
}

// 简单的节拍检测：寻找能量峰值
void detect_beats(const std::vector<double> &energy, int sample_rate, 
                  std::vector<double> &beat_times) {
    double threshold = 0.0;
    int num_frames = static_cast<int>(energy.size());
    
    // 动态计算阈值：取前10帧的平均能量作为基准
    if (num_frames > 10) {
        double sum = 0.0;
        for (int i = 0; i < 10; i++) sum += energy[i];
        threshold = (sum / 10.0) * 1.5; // 阈值设为前段平均能量的1.5倍
    } else {
        threshold = 0.01; // 默认极小阈值，防止静音文件报错
    }

    int min_distance = 5; // 最小节拍间隔（帧数）

    for (int i = 1; i < num_frames - 1; i++) {
        // 1. 必须超过阈值
        if (energy[i] > threshold) {
            // 2. 简单的峰值检测：当前能量大于前一个和后一个
            if (energy[i] > energy[i-1] && energy[i] > energy[i+1]) {
                // 3. 检查是否是局部最大值
                bool is_peak = true;
                for (int j = 1; j <= min_distance; j++) {
                    if (i - j >= 0 && energy[i] <= energy[i - j]) { is_peak = false; break; }
                    if (i + j < num_frames && energy[i] <= energy[i + j]) { is_peak = false; break; }
                }
                
                if (is_peak) {
                    // 计算时间：当前帧索引 * 每一帧的时间跨度
                    double time_sec = i * (double)HOP_SIZE / sample_rate;
                    beat_times.push_back(time_sec);
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "用法: " << argv[0] << " <音频文件.wav>" << std::endl;
        return 1;
    }

    const char *filename = argv[1];
    SF_INFO sfinfo;
    // 打开音频文件
    SNDFILE *sndfile = sf_open(filename, SFM_READ, &sfinfo);
    if (!sndfile) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return 1;
    }

    int sample_rate = sfinfo.samplerate;
    int channels = sfinfo.channels;
    sf_count_t total_frames = sfinfo.frames;
    
    std::cout << "音频加载成功。时长: " << (double)total_frames / sample_rate 
              << " 秒, 采样率: " << sample_rate 
              << " Hz, 通道数: " << channels << std::endl;

    // 预估最大可能的帧数
    int max_frames = static_cast<int>((total_frames / HOP_SIZE) + 1);

    // 使用 vector 管理内存，避免手动 malloc/free
    // 注意：sf_read_double 读取的是交错数据，如果文件是立体声，读取 HOP_SIZE 个双精度字
    // 实际上只包含了 HOP_SIZE / channels 个样本点。
    // 为了方便处理，我们读取足够多的数据填满一帧 (FRAME_SIZE 个样本点)
    int items_to_read = HOP_SIZE * channels; 
    
    std::vector<double> raw_input(items_to_read);
    std::vector<double> frame_buffer(FRAME_SIZE, 0.0); // FFT 窗口缓冲区
    std::vector<double> magnitude(FRAME_SIZE);
    std::vector<double> energy(max_frames, 0.0);
    std::vector<double> beat_times;
    
    // FFTW 仍然使用 C 风格指针分配，因为它是 C 库
    fftw_complex *fft_in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * FRAME_SIZE);
    fftw_complex *fft_out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * FRAME_SIZE);
    
    // 创建 FFTW 计划
    fftw_plan plan = fftw_plan_dft_1d(FRAME_SIZE, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);

    std::cout << "开始处理..." << std::endl;

    int energy_index = 0;

    // 主循环
    while (sf_read_double(sndfile, raw_input.data(), items_to_read) > 0) {
        // --- 1. 滑动窗口逻辑 ---
        // 旧数据左移
        std::move(frame_buffer.begin() + HOP_SIZE, frame_buffer.end(), frame_buffer.begin());
        
        // --- 2. 多声道混合降采样 (修复原 C 代码的重大 Bug) ---
        // 如果是立体声，raw_input 包含 [L1, R1, L2, R2...]，我们需要把它们混合成单声道填入 frame_buffer
        // 填入 frame_buffer 的后半部分
        for (int i = 0; i < HOP_SIZE; i++) {
            double sample = 0.0;
            // 对所有声道求平均值
            for (int ch = 0; ch < channels; ch++) {
                sample += raw_input[i * channels + ch];
            }
            frame_buffer[FRAME_SIZE - HOP_SIZE + i] = sample / channels;
        }

        // --- 3. 填充 FFT 输入 ---
        for (int i = 0; i < FRAME_SIZE; i++) {
            fft_in[i][0] = frame_buffer[i];
            fft_in[i][1] = 0.0;
        }

        // --- 4. 执行 FFT ---
        fftw_execute(plan);
        
        // --- 5. 计算幅度 ---
        calculate_magnitude(fft_out, magnitude);
        
        // --- 6. 计算能量 ---
        double frame_energy = 0.0;
        for (int i = 0; i < FRAME_SIZE / 2; i++) {
            frame_energy += magnitude[i];
        }
        
        // --- 7. 存储能量 ---
        if (energy_index < max_frames) {
            energy[energy_index] = frame_energy;
            energy_index++;
        }
    }

    // 关闭文件
    sf_close(sndfile);

    std::cout << "处理完成，共 " << energy_index << " 帧，开始检测节拍..." << std::endl;

    // 检测节拍
    detect_beats(energy, sample_rate, beat_times);
    std::cout << "共检测到 " << beat_times.size() << " 个节拍。" << std::endl;

    // 打印节拍时间
    if (!beat_times.empty()) {
        std::cout << "前20个节拍时间戳 (秒):" << std::endl;
        int print_limit = std::min((size_t)20, beat_times.size());
        for (int i = 0; i < print_limit; i++) {
            std::cout << beat_times[i] << " ";
            if ((i + 1) % 5 == 0) std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // 释放 FFTW 资源
    fftw_destroy_plan(plan);
    fftw_free(fft_in);
    fftw_free(fft_out);
    // vector 会自动释放，无需手动 free

    return 0;
}
