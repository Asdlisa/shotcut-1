// beat_tracker.cpp
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>
#include <sndfile.h>
#include <fftw3.h>

// 使用 constexpr 定义编译期常量
constexpr int BUFFER_SIZE = 1024;
constexpr int HOP_SIZE = 512;
constexpr int FRAME_SIZE = 1024;

// RAII 包装器用于 FFTW 内存，自动释放
struct FFTWVector {
    fftw_complex* data = nullptr;
    size_t size = 0;

    explicit FFTWVector(size_t n) : size(n) {
        data = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * n));
    }

    ~FFTWVector() {
        if (data) fftw_free(data);
    }

    // 禁止拷贝，允许移动
    FFTWVector(const FFTWVector&) = delete;
    FFTWVector& operator=(const FFTWVector&) = delete;
    FFTWVector(FFTWVector&&) noexcept = default;
    FFTWVector& operator=(FFTWVector&&) noexcept = default;
};

// 计算复数数组的幅度
void calculate_magnitude(const fftw_complex *in, std::vector<double> &out) {
    for (size_t i = 0; i < out.size(); i++) {
        // 使用 std::hypot 提高数值稳定性和可读性
        out[i] = std::hypot(in[i][0], in[i][1]);
    }
}

// 节拍检测：寻找能量峰值
void detect_beats(const std::vector<double> &energy, int sample_rate, 
                  std::vector<double> &beat_times) {
    if (energy.empty()) return;

    double threshold = 0.0;
    const int num_frames = static_cast<int>(energy.size());
    
    // 动态阈值：基于前 10 帧的平均能量
    if (num_frames > 10) {
        double sum = 0.0;
        for (int i = 0; i < 10; ++i) sum += energy[i];
        threshold = (sum / 10.0) * 1.5;
    } else {
        threshold = 0.01; 
    }

    constexpr int min_distance = 5; 

    for (int i = 1; i < num_frames - 1; ++i) {
        if (energy[i] > threshold) {
            // 峰值检测：大于前后邻居
            if (energy[i] > energy[i-1] && energy[i] > energy[i+1]) {
                // 局部最大值检查
                bool is_peak = true;
                for (int j = 1; j <= min_distance; ++j) {
                    if (i - j >= 0 && energy[i] <= energy[i - j]) { is_peak = false; break; }
                    if (i + j < num_frames && energy[i] <= energy[i + j]) { is_peak = false; break; }
                }
                
                if (is_peak) {
                    double time_sec = static_cast<double>(i) * HOP_SIZE / sample_rate;
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
    SF_INFO sfinfo{};
    // 打开音频文件
    SNDFILE *sndfile = sf_open(filename, SFM_READ, &sfinfo);
    if (!sndfile) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return 1;
    }

    const int sample_rate = sfinfo.samplerate;
    const int channels = sfinfo.channels;
    const sf_count_t total_frames = sfinfo.frames;
    
    std::cout << "音频加载成功。时长: " << static_cast<double>(total_frames) / sample_rate 
              << " 秒, 采样率: " << sample_rate 
              << " Hz, 通道数: " << channels << std::endl;

    // 预估帧数
    const int max_frames = static_cast<int>((total_frames / HOP_SIZE) + 1);
    const int items_to_read = HOP_SIZE * channels; 
    
    // 容器管理
    std::vector<double> raw_input(items_to_read);
    
    // --- 核心优化：环形缓冲区 ---
    // 比原代码的 std::move (O(N)) 快得多，这里是 O(1)
    std::vector<double> ring_buffer(FRAME_SIZE * 2); 
    int write_index = 0;

    std::vector<double> magnitude(FRAME_SIZE);
    std::vector<double> energy(max_frames, 0.0);
    std::vector<double> beat_times;
    
    // 使用 RAII 包装器管理 FFTW 内存
    FFTWVector fft_in(FRAME_SIZE);
    FFTWVector fft_out(FRAME_SIZE);
    
    // 创建 FFTW 计划
    // 使用 unique_ptr 确保计划在作用域结束时被销毁
    auto plan_deleter = [](fftw_plan* p){ fftw_destroy_plan(*p); delete p; };
    std::unique_ptr<fftw_plan, decltype(plan_deleter)> plan_ptr(new fftw_plan(
        fftw_plan_dft_1d(FRAME_SIZE, fft_in.data, fft_out.data, FFTW_FORWARD, FFTW_ESTIMATE)
    ), plan_deleter);
    fftw_plan plan = *plan_ptr;

    std::cout << "开始处理..." << std::endl;

    int energy_index = 0;

    // 主循环
    while (sf_read_double(sndfile, raw_input.data(), items_to_read) > 0) {
        // --- 1. 环形缓冲区写入 ---
        // 每次填入 HOP_SIZE 个新样本
        for (int i = 0; i < HOP_SIZE; ++i) {
            double sample = 0.0;
            // 多声道混合
            for (int ch = 0; ch < channels; ++ch) {
                sample += raw_input[i * channels + ch];
            }
            // 写入环形缓冲区
            ring_buffer[write_index] = sample / channels;
            ring_buffer[write_index + FRAME_SIZE] = sample / channels; // 镜像一份，方便读取
            write_index = (write_index + 1) % FRAME_SIZE;
        }

        // --- 2. 填充 FFT 输入 ---
        // 直接从环形缓冲区连续读取 FRAME_SIZE 个数据（利用镜像或模运算）
        // 这里为了演示逻辑清晰，使用简单的模运算读取（虽然比镜像稍慢，但比 std::move 快得多）
        for (int i = 0; i < FRAME_SIZE; ++i) {
            // 获取当前窗口的数据：(当前位置 - FRAME_SIZE + i)
            int read_pos = (write_index - FRAME_SIZE + i + FRAME_SIZE) % FRAME_SIZE;
            fft_in.data[i][0] = ring_buffer[read_pos];
            fft_in.data[i][1] = 0.0;
        }

        // --- 3. 执行 FFT ---
        fftw_execute(plan);
        
        // --- 4. 计算幅度 ---
        calculate_magnitude(fft_out.data, magnitude);
        
        // --- 5. 计算能量 ---
        double frame_energy = std::accumulate(magnitude.begin(), magnitude.begin() + FRAME_SIZE/2, 0.0);
        
        // --- 6. 存储能量 ---
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
        const int print_limit = std::min(static_cast<size_t>(20), beat_times.size());
        for (int i = 0; i < print_limit; ++i) {
            std::cout << beat_times[i] << " ";
            if ((i + 1) % 5 == 0) std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}
