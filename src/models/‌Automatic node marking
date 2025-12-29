// beat_tracker.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sndfile.h>
#include <fftw3.h>

#define BUFFER_SIZE 1024
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
    double threshold = 0.0;
    int min_distance = 5; // 最小节拍间隔（帧数）
    *num_beats = 0;

    for (int i = 1; i < num_frames - 1; i++) {
        // 简单的峰值检测：当前能量大于前一个和后一个
        if (energy[i] > energy[i-1] && energy[i] > energy[i+1]) {
            // 检查是否是局部最大值
            int is_peak = 1;
            for (int j = 1; j <= min_distance; j++) {
                if (i - j >= 0 && energy[i] <= energy[i - j]) { is_peak = 0; break; }
                if (i + j < num_frames && energy[i] <= energy[i + j]) { is_peak = 0; break; }
            }
            if (is_peak) {
                beat_times[*num_beats] = i * HOP_SIZE / (double)FRAME_SIZE;
                (*num_beats)++;
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
    int channels = sfinfo.channels;
    int total_frames = sfinfo.frames;
    printf("音频加载成功。时长: %.2f 秒, 采样率: %d Hz, 通道数: %d\n", 
           (double)total_frames / sample_rate, sample_rate, channels);

    // 分配内存
    double *buffer = (double *)malloc(BUFFER_SIZE * sizeof(double));
    fftw_complex *fft_in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * FRAME_SIZE);
    fftw_complex *fft_out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * FRAME_SIZE);
    double *magnitude = (double *)malloc(FRAME_SIZE * sizeof(double));
    double *energy = (double *)malloc(total_frames / HOP_SIZE * sizeof(double));
    double *beat_times = (double *)malloc(total_frames / HOP_SIZE * sizeof(double));
    int num_beats = 0;

    // 创建 FFTW 计划
    fftw_plan plan = fftw_plan_dft_1d(FRAME_SIZE, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);

    int frame_count = 0;
    int read_count;
    while ((read_count = sf_read_double(sndfile, buffer, BUFFER_SIZE)) > 0) {
        // 处理音频数据
        for (int i = 0; i < read_count; i++) {
            if (frame_count < FRAME_SIZE) {
                fft_in[frame_count][0] = buffer[i];
                fft_in[frame_count][1] = 0.0;
                frame_count++;
            }
        }

        // 当我们收集到一帧数据时
        if (frame_count == FRAME_SIZE) {
            // 执行 FFT
            fftw_execute(plan);
            
            // 计算幅度
            calculate_magnitude(fft_out, magnitude, FRAME_SIZE);
            
            // 计算能量（简化：只取前半部分，因为频谱是对称的）
            double frame_energy = 0.0;
            for (int i = 0; i < FRAME_SIZE / 2; i++) {
                frame_energy += magnitude[i];
            }
            
            // 存储能量
            energy[frame_count / HOP_SIZE] = frame_energy;
            
            // 重置 frame_count 以便下一帧
            frame_count = 0;
        }
    }

    // 关闭文件
    sf_close(sndfile);

    // 检测节拍
    detect_beats(energy, total_frames / HOP_SIZE, beat_times, &num_beats);
    printf("共检测到 %d 个节拍。\n", num_beats);

    // 打印节拍时间
    printf("节拍时间戳 (秒):\n");
    for (int i = 0; i < num_beats; i++) {
        printf("%.4f\n", beat_times[i]);
    }

    // 释放内存
    fftw_destroy_plan(plan);
    fftw_free(fft_in);
    fftw_free(fft_out);
    free(buffer);
    free(magnitude);
    free(energy);
    free(beat_times);

    return 0;
}
