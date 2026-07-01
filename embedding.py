import os
import time
from datetime import datetime

from methods import *

try:
    import tracemalloc
except ImportError:
    tracemalloc = None


def build_output_base(audio_file, payload_file, output_root='STEGOAUDIO'):
    audio_name = os.path.splitext(os.path.basename(audio_file))[0]
    payload_name = os.path.splitext(os.path.basename(payload_file))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_root, '{}_{}_{}'.format(audio_name, payload_name, timestamp))
    return os.path.join(output_dir, 'stegoaudio')


def run_single_embedding(payload_file, audio_file, total_shares, min_shares, output_base=None):
    if output_base is None:
        output_base = build_output_base(audio_file, payload_file)

    start = time.time()
    if tracemalloc is not None:
        tracemalloc.start()

    binary_payload = read_payload(payload_file)

    frame_rate, original_sample = sampling(audio_file)
    interpolated_sample = interpolation_linear(original_sample)

    bit = determine_sample_space(interpolated_sample)
    segmented_payload, last_bit = segmentation(binary_payload, bit)
    decimal_payload, isZeroInLast = convert_bin_to_dec(segmented_payload)
    prime_number = get_prime_number(decimal_payload)

    data_shares = shamir_secret_sharing(decimal_payload, prime_number, total_shares, min_shares)
    embedded = embedding(data_shares, interpolated_sample, total_shares, last_bit, isZeroInLast)

    stego_data = combine(embedded, original_sample)
    create_stego_audio(stego_data, output_base, frame_rate)

    if tracemalloc is not None:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    else:
        peak = 0
    runtime = time.time() - start

    output_files = [
        '{}{}.wav'.format(output_base, index)
        for index in range(total_shares)
    ]

    return {
        'output_base': output_base,
        'output_dir': os.path.dirname(output_base),
        'output_files': output_files,
        'runtime': runtime,
        'peak_memory_mb': peak / 10**6,
        'total_shares': total_shares,
        'min_shares': min_shares,
    }


def main():
    audio_no = input("Enter audio number from DATASET: ").strip()
    payload_no = input("Enter payload number from DATASET: ").strip()

    if not audio_no.isdigit() or not payload_no.isdigit():
        raise ValueError("Audio number and payload number must be integers.")

    audio_file = 'DATASET/Audio/data{}_mono.wav'.format(audio_no)
    payload_file = 'DATASET/Payload/payload{}.txt'.format(payload_no)
    output_base = 'STEGOAUDIO/stego_audio{}_payload{}/stegoaudio'.format(
        audio_no,
        payload_no
    )

    if not os.path.isfile(audio_file):
        raise FileNotFoundError("Audio file not found: {}".format(audio_file))
    if not os.path.isfile(payload_file):
        raise FileNotFoundError("Payload file not found: {}".format(payload_file))

    total_shares, min_shares = map(
        int,
        input("Enter total and minimum shares (n k): ").split()
    )
    if total_shares < 1 or not 1 <= min_shares <= total_shares:
        raise ValueError("Value of shares must satisfy 1 <= k <= n.")

    result = run_single_embedding(
        payload_file,
        audio_file,
        total_shares,
        min_shares,
        output_base
    )
    
    print("\nEmbedding completed successfully.")
    print("Output folder:", result['output_dir'])
    print("Peak memory:", result['peak_memory_mb'], "MB")
    print("Embedding runtime:", result['runtime'])


if __name__ == '__main__':
    main()
