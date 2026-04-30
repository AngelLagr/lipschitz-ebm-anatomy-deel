import torch as t
import torch.fft as fft
import matplotlib.pyplot as plt
FFT_NOISE = True
FFT_NOISE_BANDS = ('low', 'mid', 'high')
device = 'cuda' if t.cuda.is_available() else 'cpu'
def make_freq_mask(sz, band, device):
    y = fft.fftfreq(sz, device=device).reshape(-1, 1)
    x = fft.fftfreq(sz, device=device).reshape(1, -1)
    r2 = x**2 + y**2
    if band == 'low':
        return t.exp(-r2 / (2 * 0.04**2))
    elif band == 'mid':
        low = t.exp(-r2 / (2 * 0.04**2))
        high = t.exp(-r2 / (2 * 0.15**2))
        return high - low
    elif band == 'high':
        return 1 - t.exp(-r2 / (2 * 0.15**2))
    raise RuntimeError('Invalid band for frequency mask (use "low", "mid", or "high")')


def sample_freq_noise(batch_size, band='mixed'):
    if band == 'mixed':
        bands = [random.choice(FFT_NOISE_BANDS) for _ in range(batch_size)]
    elif band in FFT_NOISE_BANDS:
        bands = [band] * batch_size
    else:
        raise RuntimeError('Invalid band for frequency noise (use "low", "mid", "high", or "mixed")')

    noise = t.randn([batch_size, 3, 32, 32], device=device)
    freq = fft.fft2(noise)
    filtered = t.empty_like(noise)

    for current_band in FFT_NOISE_BANDS:
        current_inds = [idx for idx, band_name in enumerate(bands) if band_name == current_band]
        if not current_inds:
            continue
        mask = make_freq_mask(32, current_band, device)[None, None, :, :]
        filtered[current_inds] = fft.ifft2(freq[current_inds] * mask).real

    return filtered / (filtered.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-8)

def radial_profile(freq_tensor):
    """Calcule l'énergie en fonction du rayon (fréquence)"""
    B, C, H, W = freq_tensor.shape

    y = fft.fftfreq(H, device=freq_tensor.device).reshape(-1, 1)
    x = fft.fftfreq(W, device=freq_tensor.device).reshape(1, -1)
    r = t.sqrt(x**2 + y**2)

    r = r.flatten()
    energy = (freq_tensor.abs()**2).mean(dim=(0,1)).flatten()

    # binning radial
    bins = t.linspace(0, r.max(), steps=50)
    radial_energy = t.zeros(len(bins)-1)

    for i in range(len(bins)-1):
        mask = (r >= bins[i]) & (r < bins[i+1])
        if mask.sum() > 0:
            radial_energy[i] = energy[mask].mean()

    return bins[:-1].cpu(), radial_energy.cpu()

def analyze_band(noise):
    freq = fft.fft2(noise)
    r, e = radial_profile(freq)
    return r, e

# === TEST ===
bands = ['low', 'mid', 'high']

plt.figure(figsize=(6,4))

for band in bands:
    noise = sample_freq_noise(batch_size=32, band=band)
    r, e = analyze_band(noise)
    plt.plot(r, e / e.max(), label=band)

plt.xlabel("Fréquence (rayon)")
plt.ylabel("Énergie")
plt.legend()

plt.title("Répartition fréquentielle du bruit")
plt.savefig("freq_noise_analysis.png")
plt.show()

plt.figure(figsize=(6,4))
freq = fft.fftshift(fft.fft2(noise[0,0]))
plt.imshow(t.log(freq.abs() + 1e-8).cpu(), cmap='inferno')
plt.title(f"Spectre {band}")
plt.colorbar()
plt.savefig("specter_analysis.png")

plt.show()