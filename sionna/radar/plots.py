import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

class SpectrogramPlotter:
    """
    Classe para geração e plotagem do espectrograma de um sinal proveniente
    de um RadarSource (por exemplo, FMCWSource).

    Esta classe recebe a instância do RadarSource no construtor e utiliza
    suas propriedades para gerar e plotar o espectrograma.
    """

    def __init__(self,
                 radar_source,
                 # Parâmetros de espectrograma
                 nperseg=256,
                 noverlap=128,
                 nfft=256,
                 window='hann',
                 cmap='viridis'):
        """
        Parâmetros:
        -----------
        radar_source : RadarSource ou FMCWSource
            Instância de uma classe derivada de RadarSource que gera o sinal.
        nperseg : int
            Número de amostras em cada segmento para o cálculo do STFT.
        noverlap : int
            Sobreposição entre segmentos no STFT.
        nfft : int
            Tamanho da FFT para cada segmento.
        window : str
            Tipo de janela (e.g., 'hann', 'hamming', etc.).
        cmap : str
            Mapa de cor para o espectrograma.
        """
        # Guardamos a referência para o objeto RadarSource/FMCWSource
        self.radar_source = radar_source

        # Extraímos informações úteis do radar_source, se existirem
        # (No caso de FMCWSource, por exemplo, sample_rate, output_format etc.)
        self.sample_rate = getattr(radar_source, 'sample_rate', 1.0)
        self.output_format = getattr(radar_source, 'output_format', 'Samples')

        # Parâmetros de espectrograma
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.window = window
        self.cmap = cmap

        # Para armazenar o sinal gerado
        self.waveform = None

    def generate_signal(self, *args, **kwargs):
        """
        Gera o sinal chamando o objeto RadarSource/FMCWSource.
        Caso seja necessário, parâmetros extras podem ser passados via *args, **kwargs
        e serão repassados para a chamada do radar_source.
        
        Retorna:
        --------
        waveform : tf.Tensor
            O sinal gerado pelo radar_source.
        """
        # Chama o radar_source (que internamente executa o método call())
        # Ex.: se for FMCWSource com output_format='Samples', podemos passar
        #     num_samples como argumento, etc.
        # Se não houver argumentos, passamos um dummy input.
        if not args and "inputs" not in kwargs:
            dummy_input = tf.constant(0)
            self.waveform = self.radar_source(dummy_input, *args, **kwargs)
        else:
            self.waveform = self.radar_source(*args, **kwargs)
        return self.waveform

    def plot_spectrogram(self):
        """
        Plota o espectrograma do sinal armazenado em self.waveform.
        Se o sinal ainda não tiver sido gerado, chama generate_signal() sem parâmetros.
        """
        if self.waveform is None:
            # Gera sem parâmetros extras, usando defaults
            self.generate_signal()

        # Se o radar_source estiver configurado para gerar 'Sweeps' (2D), podemos achatar:
        # Formato [num_sweeps, num_amostras_por_sweep]
        if self.output_format.lower() == 'sweeps':
            signal_1d = tf.reshape(self.waveform, [-1])
        else:
            signal_1d = self.waveform  # 'Samples' já é 1D

        # Convertemos para numpy para usar scipy.signal.spectrogram
        sig_np = signal_1d.numpy()

        # Podemos analisar o sinal complexo direto ou a magnitude. Exemplos:
        #    x = np.abs(sig_np)
        #    x = sig_np.real
        #    x = sig_np
        # Abaixo, faremos a análise com a magnitude:
        x = np.real(sig_np)

        # Calcula o espectrograma
        f, t_spec, Sxx = spectrogram(
            x=x,
            fs=self.sample_rate,
            window=self.window,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft,
            scaling='density'  # densidade espectral (potência por Hz)
        )

        # Converte para escala dB (adicionando um pequeno offset para evitar log(0))
        Sxx_dB = 10 * np.log10(Sxx + 1e-12)

        # Plot
        plt.figure(figsize=(8, 4))
        plt.pcolormesh(t_spec/1e6, f/1e6, Sxx_dB, shading='gouraud', cmap=self.cmap)
        plt.title("Espectrograma do Sinal Radar")
        plt.xlabel("Tempo [us]")
        plt.ylabel("Frequência [MHz]")
        cbar = plt.colorbar()
        cbar.set_label("Intensidade [dB/Hz]")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import tensorflow as tf
    from sionna.radar.source import FMCWSource

    # 1) Criar a instância do FMCWSource
    fmcw_source = FMCWSource(
        sample_rate=1e6,
        sweep_time=7e-6,
        sweep_bandwidth=1e5,
        sweep_direction='Up',
        sweep_interval='Positive',
        output_format='Samples',
        num_samples=2000  # se for 'Samples'
    )

    # 2) Criar a instância do SpectrogramPlotter, passando o fmcw_source
    sp = SpectrogramPlotter(
        radar_source=fmcw_source,
        nperseg=256,
        noverlap=128,
        nfft=256,
        window='hann',
        cmap='viridis'
    )

    # 3) Gerar o sinal (opcionalmente, poderíamos passar parâmetros extras)
    waveform = sp.generate_signal()

    # waveform agora está disponível em sp.waveform
    print("Forma do waveform gerado:", waveform.shape)

    # 4) Plotar o espectrograma
    sp.plot_spectrogram()
