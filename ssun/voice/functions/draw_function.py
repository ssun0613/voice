import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def tensorboard_draw(writer, mel_in, mel_outs, mel_loss, global_step):

    writer.add_scalar("loss/loss", mel_loss, global_step)
    target = []
    prediction_FC1 = []
    prediction_FC2 = []
    prediction_FC3 = []

    for i in range(2):
        target_spectogram = (mel_in[i].squeeze(0).permute(1, 0).cpu().detach().numpy())
        target_spectogram = plot_spectrogram(target_spectogram)
        target.append(target_spectogram)

        FC1 = (mel_outs[0][i].squeeze(0).permute(1, 0).cpu().detach().numpy())
        FC1 = plot_spectrogram(FC1)
        prediction_FC1.append(FC1)

        FC2 = (mel_outs[1][i].squeeze(0).permute(1, 0).cpu().detach().numpy())
        FC2 = plot_spectrogram(FC2)
        prediction_FC2.append(FC2)

        FC3 = (mel_outs[2][i].squeeze(0).permute(1, 0).cpu().detach().numpy())
        FC3 = plot_spectrogram(FC3)
        prediction_FC3.append(FC3)

    writer.add_figure('mel-spectrogram/target', target, global_step)
    writer.add_figure('mel-spectrogram/prediction_FC1', prediction_FC1, global_step)
    writer.add_figure('mel-spectrogram/prediction_FC2', prediction_FC2, global_step)
    writer.add_figure('mel-spectrogram/prediction_FC3', prediction_FC3, global_step)

def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig
