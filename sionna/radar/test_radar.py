from sionna.radar.source import FMCWSource
from sionna.radar.plots import SpectrogramPlotter, TimeDomainPlotter

def main():
    # Configurações do FMCW
    sample_rate = 20.0e6  # Taxa de amostragem de 20 MHz
    sweep_time = 1e-4     # Tempo de varredura de 100 μs
    sweep_bandwidth = 10.0e6  # Largura de banda de varredura de 10 MHz
    sweep_direction = 'UP'  # Direção da varredura triangular
    sweep_interval = 'Positive'  # Intervalo de varredura positivo [0, B]
    output_format = 'Sweeps'  # Formato de saída: múltiplas varreduras
    num_sweeps = 2  # Número de varreduras

    # Criar a instância do FMCWSource
    fmcw_source = FMCWSource(
        sample_rate=sample_rate,
        sweep_time=sweep_time,
        sweep_bandwidth=sweep_bandwidth,
        sweep_direction=sweep_direction,
        sweep_interval=sweep_interval,
        output_format=output_format,
        num_sweeps=num_sweeps
    )

    # Criar a instância do TimeDomainPlotter
    time_plotter = TimeDomainPlotter(fmcw_source)

    # Gerar o sinal FMCW
    waveform = time_plotter.generate_signal()

    # Plotar o sinal no domínio do tempo
    time_plotter.plot_time_domain(
        signal_component='real',
        figsize=(10, 6),
        title="Sinal FMCW no Domínio do Tempo",
        xlabel="Tempo [us]",
        ylabel="Amplitude",
        save_path="time_domain_plot.png"
    )

    # Criar a instância do SpectrogramPlotter
    spectrogram_plotter = SpectrogramPlotter(
        radar_source=fmcw_source,
        nperseg=256,  # Número de amostras por segmento
        noverlap=128,  # Sobreposição entre segmentos
        nfft=256,  # Tamanho da FFT
        window='hann',  # Tipo de janela
        cmap='viridis'  # Mapa de cores
    )

    # Plotar o espectrograma
    spectrogram_plotter.plot_spectrogram(
        time_scale=3.0,
        signal_component='real',
        figsize=(10, 6),
        title="Espectrograma do Sinal FMCW",
        xlabel="Tempo [us]",
        ylabel="Frequência [MHz]"
    )

if __name__ == "__main__":
    main()