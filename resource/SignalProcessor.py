import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import welch, csd
from scipy.fft import fft
from scipy.signal import butter, lfilter
import control as ctrl
from control.matlab import * 

# 信号処理を行うクラス
class SignalProcessor:
    def __init__(self, input_data, output_data, fs, cutoff_freq, time):
        self.input_data = input_data  # 入力データ
        self.output_data = output_data  # 出力データ
        self.fs = fs  # サンプリング周波数
        self.cutoff_freq = cutoff_freq # カットオフ周波数 (Hz)
        self.time = time # 読み取った録音データの時間

    def compute_optimal_v(self, n, u, y):
        # データ数の取得
        N = len(y)-1

        # Y の構築
        Y = np.array(y[n:]).reshape(-1, 1)

        # X_1 の構築
        X_1 = np.zeros((N-n+1, n))
        for i in range(N-n+1):
            for j in range(n):
                X_1[i, j] = y[n-1+i-j]

        # X_2 の構築
        X_2 = np.zeros((N-n+1, n+1))
        for i in range(N-n+1):
            for j in range(n+1):
                X_2[i, j] = u[n+i-j]
        
        # X の構築
        X = np.hstack((X_1, X_2))

        # 行列が正則かチェック
        if np.linalg.det(X.T @ X) == 0:
            raise ValueError("行列が正則ではありません。")

        # 最適化問題を解く
        v = np.linalg.inv(X.T @ X) @ X.T @ Y

        # vからv1とv2を抽出
        v1 = v[:n]
        v2 = v[n:]

        # 係数ベクトルaの生成
        a = np.zeros(n+1)
        for i in range(n+1):
            if i==0:
                a[i] = 1
            else:
                a[i] = v1[i-1]

        # 係数ベクトルbの生成
        b = np.zeros(n+1)
        for i in range(n+1):
            b[i] = v2[i]

        return a, b

    def identify_system(self, n):
        """
        システム同定
        """
        Dp, Np = self.compute_optimal_v(n, self.input_data, self.output_data)
        # Np: 伝達関数の分子多項式の係数
        # Dp: 伝達関数の分母多項式の係数
        P = ctrl.tf(Np, Dp)

        ts = 1/self.fs # サンプリング時間
        T = np.arange(0, self.time, ts)

        # 加工したデータの出力
        processed_data, _, _ = ctrl.matlab.lsim(P, self.input_data, T)

        return processed_data

    def plot_fft_spectrum(self, signal, sampling_rate=2000):
        # FFTスペクトルをプロットする関数
        N = len(signal)  # サンプル点の数

        # FFTと周波数領域
        yf = fft(signal)
        xf = np.linspace(0.0, sampling_rate/2, N//2)

        # プロット
        plt.figure(figsize=(10, 6))
        plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
        plt.title("FFT Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

    def apply_low_pass_filter(self, signal, cutoff_frequency, sampling_rate=2000):
        # ローパスフィルタを適用する関数
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_frequency / nyquist

        # Butterworthフィルタを設計
        b, a = butter(N=1, Wn=normal_cutoff, btype='low', analog=False)

        # フィルタを適用
        filtered_signal = lfilter(b, a, signal)
        return filtered_signal

    def compute_power_spectrum(self, data):
        # パワースペクトルを計算する関数
        f, Pxx = welch(data, fs=self.fs)
        return f, Pxx

    def compute_cross_spectrum(self, input, output):
        # クロススペクトルを計算する関数
        f, Pxy = csd(input, output, fs=self.fs)
        return f, Pxy

    def compute_frequency_response_function(self, output):
        # 周波数応答関数を計算する関数
        f, Pxx = self.compute_power_spectrum(self.input_data)
        _, Pxy = self.compute_cross_spectrum(self.input_data, self.output_data)
        H = Pxy / Pxx
        _, Pxx = self.compute_power_spectrum(self.input_data)
        _, Pxy = self.compute_cross_spectrum(self.input_data, output)
        P = Pxy / Pxx
        return f, H, P

    def plot_bode(self, f, H, P):
        # ボード線図（ゲインと位相のグラフ）を描画する関数
        gain_h = 20 * np.log10(np.abs(H))  # ゲインをデシベル単位で計算
        phase_h = np.angle(H, deg=True)  # 位相を度単位で計算

        self.plot_fft_spectrum(gain_h)
        self.plot_fft_spectrum(phase_h)

        gain_h = self.apply_low_pass_filter(gain_h, self.cutoff_freq)
        phase_h = self.apply_low_pass_filter(phase_h, self.cutoff_freq)

        # P について
        gain_p = 20 * np.log10(np.abs(P))  # ゲインをデシベル単位で計算
        phase_p = np.angle(P, deg=True)  # 位相を度単位で計算

        self.plot_fft_spectrum(gain_p)
        self.plot_fft_spectrum(phase_p)

        gain_p = self.apply_low_pass_filter(gain_p, self.cutoff_freq)
        phase_p = self.apply_low_pass_filter(phase_p, self.cutoff_freq)

        # 差の計算
        gain = gain_h - gain_p
        phase = phase_h - phase_p

        # Hのボード線図を表示したい場合
        # gain = gain_h
        # phase = phase_h

        # 同定したシステムのボード線図を表示したい場合
        # gain = gain_p
        # phase = phase_p

        # カスタムフォーマッタの定義
        def custom_formatter(x, pos):
            if x >= 1000:
                return '{:.0f}k'.format(x / 1000)
            else:
                return '{:.0f}'.format(x)

        # ゲイン線図を描画
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.semilogx(f, gain, base=2)
        plt.title('Bode Plot')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Gain [dB]')
        plt.xlim(20, 20000)
        plt.grid(which='both', linestyle='-', linewidth='0.5')
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
        plt.gca().xaxis.set_minor_formatter(ticker.FuncFormatter(custom_formatter))

        # 位相線図を描画
        plt.subplot(2, 1, 2)
        plt.semilogx(f, phase, base=2)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Phase [degrees]')
        plt.xlim(20, 20000)
        plt.grid(which='both', linestyle='-', linewidth='0.5')
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
        plt.gca().xaxis.set_minor_formatter(ticker.FuncFormatter(custom_formatter))

        plt.tight_layout()
        plt.show()

    def find_flat_gain(self, f, H, P):
        # 平坦な部分のゲインを見つける関数
        gain_h = 20 * np.log10(np.abs(H))
        gain_h = self.apply_low_pass_filter(gain_h, self.cutoff_freq)
        # P について
        gain_p = 20 * np.log10(np.abs(P))
        gain_p = self.apply_low_pass_filter(gain_p, self.cutoff_freq)
        # 差の計算
        gain = gain_h - gain_p
        # 平坦な部分を特定するための簡易アルゴリズム（例えば標準偏差が小さい区間）
        std_gain = np.convolve(gain, np.ones(10)/10, mode='valid')  # 移動平均に基づく標準偏差
        flat_idx = np.argmin(np.abs(std_gain))  # 最も平坦な部分
        return f[flat_idx], gain[flat_idx]

    def find_peak_gains(self, f, H, P):
        # 平坦な部分のゲインを見つける関数
        gain_h = 20 * np.log10(np.abs(H))
        gain_h = self.apply_low_pass_filter(gain_h, self.cutoff_freq)
        # P について
        gain_p = 20 * np.log10(np.abs(P))
        gain_p = self.apply_low_pass_filter(gain_p, self.cutoff_freq)
        # 差の計算
        gain = gain_h - gain_p
        peaks, _ = find_peaks(gain, height=0)  # 山の頂点を見つける
        # ゲインが高い順に並べ替え
        peak_gains = sorted(zip(f[peaks], gain[peaks]), key=lambda x: x[1], reverse=True)
        return peak_gains[:3]  # 最大3つの山

    def gain_difference(self, f, H, P):
        # 平坦な部分と山の頂点のゲインの差を計算する関数
        _, flat_gain = self.find_flat_gain(f, H, P)
        peak_gains = self.find_peak_gains(f, H, P)
        differences = [(pf, pg - flat_gain) for pf, pg in peak_gains]
        return differences

    def write_results_to_file(self, filepath, f, H, P):
        with open(filepath, 'w') as file:
            # 平坦なゲインの値を書き出す
            flat_freq, flat_gain = self.find_flat_gain(f, H, P)
            file.write(f"平坦なゲインの周波数: {flat_freq:.2f} Hz, ゲイン: {flat_gain:.2f} dB\n")

            # 山の頂点のゲインの値を書き出す
            peak_gains = self.find_peak_gains(f, H, P)
            file.write("山の頂点のゲインの値（上位3つ）:\n")
            for freq, gain in peak_gains:
                file.write(f"  周波数: {freq:.2f} Hz, ゲイン: {gain:.2f} dB\n")

            # ゲインの差を書き出す
            gain_diff = self.gain_difference(f, H, P)
            file.write("ゲインの差（平坦な部分と山の頂点）:\n")
            for freq, diff in gain_diff:
                file.write(f"  周波数: {freq:.2f} Hz, ゲインの差: {diff:.2f} dB\n")
