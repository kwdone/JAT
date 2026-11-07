"""Plot per-epoch val and test accuracy/CE for the released entropy run (human+animal -> cartoon).

Numbers transcribed from
results/JAT/multi_source/video_audio/['human', 'animal']_to_['cartoon']_seed0_wm_entropy.csv
"""
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EPOCHS = list(range(1, 21))
VAL_CE = [0.31698, 0.37326, 0.28102, 0.19850, 0.27427, 0.26115, 0.26105, 0.24663,
          0.21659, 0.28766, 0.27123, 0.34941, 0.24298, 0.23172, 0.26110, 0.28665,
          0.29801, 0.27837, 0.30314, 0.20516]
VAL_ACC = [0.88938, 0.89381, 0.90487, 0.93142, 0.90929, 0.92920, 0.92035, 0.93584,
           0.92699, 0.90929, 0.92478, 0.90265, 0.92920, 0.93805, 0.91814, 0.91150,
           0.92257, 0.92920, 0.92478, 0.93805]
VAL_EMA = [0.63496, 0.86726, 0.91593, 0.93363, 0.93363, 0.93584, 0.93363, 0.94248,
           0.94690, 0.94912, 0.95133, 0.95354, 0.95133, 0.95575, 0.94912, 0.95133,
           0.94912, 0.94690, 0.94469, 0.94469]
TGT_CE = [1.84863, 1.63023, 1.44793, 1.34910, 1.26800, 1.23777, 1.23955, 1.29360,
          1.34544, 1.40733, 1.44456, 1.47066, 1.51639, 1.58506, 1.64901, 1.65110,
          1.66641, 1.69875, 1.76325, 1.83359]
TGT_ACC = [0.22151, 0.36397, 0.43107, 0.48254, 0.52665, 0.55790, 0.56985, 0.56985,
           0.56434, 0.55790, 0.55515, 0.56250, 0.56710, 0.56434, 0.56066, 0.56250,
           0.55607, 0.55974, 0.55790, 0.55607]

BEST_EPOCH = 14          # val-EMA argmax (released rule)
TARGET_PEAK = 7          # first epoch at max target accuracy
ROW, COL = 1, 2

fig, (ax_acc, ax_ce) = plt.subplots(ROW, COL, figsize=(12.5, 4.6))

# ---- accuracy -------------------------------------------------------------
ax_acc.plot(EPOCHS, VAL_ACC, color='0.55', ls=':', lw=1.6, marker='.',
            label='source val, student')
ax_acc.plot(EPOCHS, VAL_EMA, color='#1f77b4', lw=2.0, marker='o', ms=3.5,
            label='source val, EMA (selection signal)')
ax_acc.plot(EPOCHS, TGT_ACC, color='#d62728', lw=2.0, marker='s', ms=3.5,
            label='target test, EMA (logged each epoch)')
ax_acc.axvline(BEST_EPOCH, color='#1f77b4', ls='--', lw=1.0, alpha=0.7)
ax_acc.axvline(TARGET_PEAK, color='#d62728', ls='--', lw=1.0, alpha=0.7)
ax_acc.annotate(f'selected ep{BEST_EPOCH}\nval {VAL_EMA[BEST_EPOCH - 1]:.3f}\n'
                f'test {TGT_ACC[BEST_EPOCH - 1]:.3f}',
                xy=(BEST_EPOCH, VAL_EMA[BEST_EPOCH - 1] - 0.012),
                xytext=(8.3, 0.70), fontsize=8, color='#1f77b4',
                ha='left',
                arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=0.8))
ax_acc.annotate(f'target peak ep{TARGET_PEAK}\n{TGT_ACC[TARGET_PEAK - 1]:.3f}',
                xy=(TARGET_PEAK, TGT_ACC[TARGET_PEAK - 1]),
                xytext=(TARGET_PEAK + 0.4, 0.30), fontsize=8, color='#d62728',
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=0.8))
ax_acc.set_xlabel('epoch')
ax_acc.set_ylabel('accuracy')
ax_acc.set_title('Accuracy: source validation rises, target test peaks at epoch 7')
ax_acc.set_xlim(1, 20)
ax_acc.set_ylim(0.20, 1.00)
ax_acc.grid(alpha=0.3)
ax_acc.legend(loc='lower right', fontsize=8, framealpha=0.9)

# ---- cross entropy --------------------------------------------------------
ax_ce.plot(EPOCHS, VAL_CE, color='#1f77b4', lw=2.0, marker='o', ms=3.5,
           label='source val CE')
ax_ce.plot(EPOCHS, TGT_CE, color='#d62728', lw=2.0, marker='s', ms=3.5,
           label='target test CE')
ax_ce.axvline(BEST_EPOCH, color='#1f77b4', ls='--', lw=1.0, alpha=0.7)
ax_ce.axvline(TARGET_PEAK, color='#d62728', ls='--', lw=1.0, alpha=0.7)
ax_ce.annotate(f'target CE minimum ep6\n{TGT_CE[5]:.3f}',
               xy=(6, TGT_CE[5]), xytext=(7.2, 0.62), fontsize=8, color='#d62728',
               arrowprops=dict(arrowstyle='->', color='#d62728', lw=0.8))
ax_ce.annotate(f'epoch 20\n{TGT_CE[-1]:.3f} (+48% vs min)', xy=(20, TGT_CE[-1]),
               xytext=(12.6, 1.66), fontsize=8, color='#d62728',
               arrowprops=dict(arrowstyle='->', color='#d62728', lw=0.8))
ax_ce.set_xlabel('epoch')
ax_ce.set_ylabel('cross entropy')
ax_ce.set_title('Loss: source CE flat, target CE degrades monotonically after epoch 6')
ax_ce.set_xlim(1, 20)
ax_ce.grid(alpha=0.3)
ax_ce.legend(loc='center right', fontsize=8, framealpha=0.9)

fig.suptitle('JAT, human+animal -> cartoon, video+audio, seed 0 '
             '(released entropy weighting, 20 epochs, bsz 8)', fontsize=10)
for ax in (ax_acc, ax_ce):
    ax.set_xticks(range(2, 21, 2))
    ax.tick_params(labelsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.95))

out = pathlib.Path(__file__).resolve().parent / 'entropy_run_val_vs_test.png'
fig.savefig(out, dpi=170)
print('saved', out)

print('\nep  val_acc  val_ema  tgt_acc   val_ce  tgt_ce')
for i, e in enumerate(EPOCHS):
    print(f'{e:2d}  {VAL_ACC[i]:.4f}   {VAL_EMA[i]:.4f}   {TGT_ACC[i]:.4f}   '
          f'{VAL_CE[i]:.4f}  {TGT_CE[i]:.4f}')
