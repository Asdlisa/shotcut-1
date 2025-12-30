// beat_tracker.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sndfile.h>
#include <fftw3.h>
#include <string.h> // 需要引入 memmove 和 memcpy

#define BUFFER_SIZE 1024  // 这里实际上我们使用 HOP_SIZE 作为读取单位
#define HOP_SIZE 512
#define FRAME_SIZE 1024

// 计算一个复数数组的幅度
void calculate_magnitude(fftw_complex *in, double *out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = sqrt(in[i][0] * in[i][0] + in[i][1] * in[i][1]);
    }
}

// 简单的节拍检测：寻找能量峰值
void detect_beats(double *energy, int num_frames, double *beat_times, int *num_beats) {
    // 设置一个稍微高一点的阈值，避免静音噪音
    // 简单的做法是：取前10帧的平均能量作为基准，然后乘以一个系数
    double threshold = 0.0;
    if (num_frames > 10) {
        double sum = 0.0;
        for (int i = 0; i < 10; i++) sum += energy[i];
        threshold = (sum / 10.0) * 1.5; // 阈值设为前段平均能量的1.5倍
    }

    int min_distance = 5; // 最小节拍间隔（帧数）
    *num_beats = 0;

    for (int i = 1; i < num_frames - 1; i++) {
        // 1. 必须超过阈值
        if (energy[i] > threshold) {
            // 2. 简单的峰值检测：当前能量大于前一个和后一个
            if (energy[i] > energy[i-1] && energy[i] > energy[i+1]) {
                // 3. 检查是否是局部最大值（防止在同一个大波峰上检测到多个点）
                int is_peak = 1;
                for (int j = 1; j <= min_distance; j++) {
                    if (i - j >= 0 && energy[i] <= energy[i - j]) { is_peak = 0; break; }
                    if (i + j < num_frames && energy[i] <= energy[i + j]) { is_peak = 0; break; }
                }
                if (is_peak) {
                    // 计算时间：当前帧索引 * 每一帧的时间跨度
                    // 每一帧的时间跨度 = HOP_SIZE / 采样率
                    beat_times[*num_beats] = i * (double)HOP_SIZE / sample_rate;
                    (*num_beats)++;
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "用法: %s <音频文件.wav>\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    SF_INFO sfinfo;
    SNDFILE *sndfile = sf_open(filename, SFM_READ, &sfinfo);
    if (!sndfile) {
        fprintf(stderr, "无法打开文件: %s\n", filename);
        return 1;
    }

    int sample_rate = sfinfo.samplerate;
    // int channels = sfinfo.channels; // 简单起见，这里主要处理单声道或自动混合
    int total_frames = sfinfo.frames;
    printf("音频加载成功。时长: %.2f 秒, 采样率: %d Hz, 通道数: %d\n", 
           (double)total_frames / sample_rate, sample_rate, sfinfo.channels);

    // 预估最大可能的帧数 (总样本数 / HOP_SIZE) + 1
    int max_frames = (total_frames / HOP_SIZE) + 1;

    // 分配内存
    // 1. 临时读取缓冲区，每次只读 HOP_SIZE 个样本
    double *input_buffer = (double *)malloc(HOP_SIZE * sizeof(double));
    
    // 2. FFT 完整帧缓冲区 (用于保持滑动窗口)
    double *frame_buffer = (double *)calloc(FRAME_SIZE, sizeof(double));
    
    fftw_complex *fft_in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * FRAME_SIZE);
    fftw_complex *fft_out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * FRAME_SIZE);
    double *magnitude = (double *)malloc(FRAME_SIZE * sizeof(double));
    
    // 3. 能量和节拍时间数组
    double *energy = (double *)calloc(max_frames, sizeof(double));
    double *beat_times = (double *)malloc(max_frames * sizeof(double));
    
    int num_beats = 0;
    int energy_index = 0; // 记录实际写入了多少帧能量数据

    // 创建 FFTW 计划
    fftw_plan plan = fftw_plan_dft_1d(FRAME_SIZE, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);

    printf("开始处理...\n");

    // 主循环：每次读取 HOP_SIZE 个样本
    while (sf_read_double(sndfile, input_buffer, HOP_SIZE) > 0) {
        // --- 核心修正：滑动窗口逻辑 ---
        
        // 1. 将旧数据向左移动：丢弃最左边的 HOP_SIZE 个，把后面的移过来
        memmove(frame_buffer, frame_buffer + HOP_SIZE, HOP_SIZE * sizeof(double));
        
        // 2. 将新读取的数据复制到最右边的空位
        memcpy(frame_buffer + HOP_SIZE, input_buffer, HOP_SIZE * sizeof(double));

        // 3. 填充 FFT 输入 (取单声道或左声道)
        for (int i = 0; i < FRAME_SIZE; i++) {
            // 注意：如果原始音频是多声道，sf_read_double 会交错输出数据 (L R L R...)
            // 简单的降采样处理：只取每个点的值，或者 (L+R)/2，这里简化为直接读取
            // 如果是立体声，索引 i 实际上可能对应 i*channels。这里假设单声道或取值。
            // 为了严格对应声道，应该乘 channels，但为保持代码结构简单，这里假设输入是单声道数据流
            // 或者你可以改为: frame_buffer[i * sfinfo.channels]
            
            fft_in[i][0] = frame_buffer[i];
            fft_in[i][1] = 0.0;
        }

        // 4. 执行 FFT
        fftw_execute(plan);
        
        // 5. 计算幅度
        calculate_magnitude(fft_out, magnitude, FRAME_SIZE);
        
        // 6. 计算能量（简化：只取前半部分）
        double frame_energy = 0.0;
        for (int i = 0; i < FRAME_SIZE / 2; i++) {
            frame_energy += magnitude[i];
        }
        
        // 7. 存储能量
        if (energy_index < max_frames) {
            energy[energy_index] = frame_energy;
            energy_index++;
        }
    }

    // 关闭文件
    sf_close(sndfile);
    free(input_buffer); // 及时释放

    printf("处理完成，共 %d 帧，开始检测节拍...\n", energy_index);

    // 检测节拍 (传入实际处理的帧数 energy_index)
    detect_beats(energy, energy_index, beat_times, &num_beats);
    printf("共检测到 %d 个节拍。\n", num_beats);

    // 打印节拍时间
    if (num_beats > 0) {
        printf("前20个节拍时间戳 (秒):\n");
        int print_limit = num_beats < 20 ? num_beats : 20;
        for (int i = 0; i < print_limit; i++) {
            printf("%.4f ", beat_times[i]);
            if ((i + 1) % 5 == 0) printf("\n");
        }
        printf("\n");
    }

    // 释放内存
    fftw_destroy_plan(plan);
    fftw_free(fft_in);
    fftw_free(fft_out);
    free(frame_buffer);
    free(magnitude);
    free(energy);
    free(beat_times);

    return 0;
}
