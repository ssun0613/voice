import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt


def tensorboard_draw(writer, mel_in, mel_out, recon_voice_loss, recon_pitch_loss, total_loss_g, total_loss_d, global_step):

    writer.add_scalar("loss/recon_voice_loss", recon_voice_loss, global_step)
    writer.add_scalar("loss/recon_pitch_loss", recon_pitch_loss, global_step)
    writer.add_scalar("loss/total_loss_g", total_loss_g, global_step)
    writer.add_scalar("loss/total_loss_d", total_loss_d, global_step)

    spectrogram_target = []
    spectrogram_prediction = []

    for i in range(mel_in.shape[0]):
        target_spectogram = (mel_in[i].squeeze(0).permute(1, 0).cpu().detach().numpy())
        target_spectogram = plot_spectrogram(target_spectogram)
        spectrogram_target.append(target_spectogram)

        prediction_spectogram = (mel_out[i].squeeze(0).permute(1, 0).cpu().detach().numpy())
        prediction_spectogram = plot_spectrogram(prediction_spectogram)
        spectrogram_prediction.append(prediction_spectogram)

    writer.add_figure('mel-spectrogram/voice_target', spectrogram_target, global_step)
    writer.add_figure('mel-spectrogram/voice_prediction', spectrogram_prediction, global_step)
def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig
