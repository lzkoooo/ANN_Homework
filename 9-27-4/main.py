# -*- coding = utf-8 -*-
# @Time : 2024/9/26 下午9:44
# @Author : 李兆堃
# @File : 4-6.py
# @Software : PyCharm

import numpy as np
import matplotlib.pyplot as plt
from Adaline import Adaline
import sounddevice as sd


def generated_data():
    # 参数设置
    # fs = 44100  # 采样率
    duration = 50  # 持续时间（秒）
    time = np.arange(0.5, duration + 0.005, 0.005)  # 时间数组

    # 生成自己声音信号
    my_voice = (np.random.rand(len(time)) - 0.5) * 2  # 随机噪声信号模拟

    # 生成噪声源（正弦波）
    frequency = 1  # 频率（赫兹）
    other_voice = 0.5 * np.sin(2 * np.pi * frequency * time)  # 噪声信号（正弦波）

    # 生成混合信号
    mix_voice = my_voice + other_voice  # 采集到的混合了噪声的 EEG 信号

    # 准备输入和目标
    input_data = [other_voice[:-2].reshape(-1, 1)]  # 输入信号
    target_data = mix_voice[2:].reshape(-1, 1)  # 目标信号

    return input_data, target_data, my_voice, time


if __name__ == '__main__':
    train_X, train_Y, my_voice, time = generated_data()
    data = []
    ori_data = []
    for i in range(len(time) - 3):
        data.append([train_X[0][i][0], train_X[0][i+1][0], train_Y[i][0]])
        ori_data.append([my_voice[i], my_voice[i+1]])
    np.savetxt('9-27-4-train.txt', np.array(data))
    np.savetxt('9-27-4-ori_data.txt', np.array(ori_data))
    W = 4 * np.random.rand(len(train_X[0]), len(train_Y)).astype(np.float64) - 2
    eta = 0.3
    max_epoch = 20

    model = Adaline(train_X, W, train_Y, eta, 'linear', max_epoch)
    model.fit()
    pred, e = model.predict(train_X)

    # # 绘制结果
    plt.figure(figsize=(12, 8))
    plt.text(0, 1.03, 'not use delay', fontsize=30, color='red', zorder=3)

    # 绘制混合信号
    plt.subplot(4, 1, 1)
    plt.plot(time[:-2], train_Y, 'k-', zorder=1)
    plt.title('mix_voice (target_data)')
    plt.xlabel('time (s)')
    plt.ylabel('range')

    # 绘制噪声信号
    plt.subplot(4, 1, 2)
    plt.plot(time[:-2], train_X[0], 'b', zorder=1)
    plt.title('other_voice (input_data)')
    plt.xlabel('time (s)')
    plt.ylabel('range')

    # 绘制预测信号
    plt.subplot(4, 1, 3)
    plt.plot(time[:-2], pred, 'y', zorder=1)
    plt.title('pred')
    plt.xlabel('time (s)')
    plt.ylabel('range')

    # 绘制消除噪声信号
    plt.subplot(4, 1, 4)
    plt.plot(time[:-2], e, 'g', zorder=1)
    plt.title('filter (error)')
    plt.xlabel('time (s)')
    plt.ylabel('range')

    plt.tight_layout()
    plt.savefig('result.png', dpi=300)
    plt.show()


