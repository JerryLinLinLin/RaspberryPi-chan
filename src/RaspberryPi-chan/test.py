import sounddevice as sd

def test_sample_rates(device, rates):
    supported_rates = []
    for rate in rates:
        try:
            # Attempt to open a raw input stream with the specified sample rate
            with sd.RawInputStream(samplerate=rate, device=device, channels=1):
                print(f"Sample rate {rate} Hz supported.")
                supported_rates.append(rate)
        except Exception as e:
            # print(f"Sample rate {rate} Hz NOT supported. Error: {e}")
            pass
    return supported_rates

# List of common sample rates to test
sample_rates = [8000, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 192000]

# Replace 'device' with the index of your input device
device_index = 0  # You might need to change this to your specific device index

# Check which sample rates are supported by your device
supported_sample_rates = test_sample_rates(device_index, sample_rates)
