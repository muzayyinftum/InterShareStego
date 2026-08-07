import os
import runpy
import time
from datetime import datetime

from methods import *

try:
    import tracemalloc
except ImportError:
    tracemalloc = None


MENU = {
    "1": ("Embedding", None),
    "2": ("Extracting", "extracting"),
    "3": ("Evaluation", None),
    "4": ("Exit", None),
}

MENU_ORDER = ("1", "2", "3", "4")

EVALUATION_MENU = {
    "1": ("Compare payload and audio", "single_compare"),
    "2": ("Check audio quality", "quality_evaluation"),
    "3": ("Detector ML", "detector_ml"),
    "4": ("NC and entropy analysis (single)", "nc_entropy_single"),
    "5": ("NC and entropy analysis (batch + excel)", "nc_entropy_batch"),
}

EVALUATION_MENU_ORDER = ("1", "2", "3", "4", "5")


def show_menu():
    print("\n" + "=" * 70)
    print("  STEGOSHARE - Reversible Audio Steganography Using Secret Sharing")
    print("=" * 70)

    for choice in MENU_ORDER:
        name, _ = MENU[choice]
        print("{}. {}".format(choice, name))


def show_menu_evaluation():
    print("\n" + "=" * 70)
    print("  EVALUATION")
    print("=" * 70)

    for choice in EVALUATION_MENU_ORDER:
        name, _ = EVALUATION_MENU[choice]
        print("{}. {}".format(choice, name))


def run_modul(name_modul):
    runpy.run_module(name_modul, run_name="__main__")


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


def run_embedding():
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


def run_detector_ml():
    from steganalysis import build_cover_path, run_detector_experiment

    print("=" * 70)
    print("  DETECTOR-BASED STEGANALYSIS (ML Benchmark)")
    print("=" * 70)

    print("Select stego_audio[X]_payload[Y] to analyze:")
    audio_no = input("Enter audio number [X]: ").strip() or "1"
    payload_no = input("Enter payload number [Y]: ").strip() or "1"
    if not audio_no.isdigit() or not payload_no.isdigit():
        raise ValueError("Audio number and payload number must be integers.")

    cover_path = build_cover_path(audio_no)
    run_detector_experiment(cover_path)


def run_nc_entropy_single():
    from steganalysis import build_cover_path, run_single

    print("=" * 70)
    print("  NC AND ENTROPY ANALYSIS (single pair)")
    print("=" * 70)

    print("Select stego_audio[X]_payload[Y] to analyze:")
    audio_no = input("Enter audio number [X]: ").strip() or "1"
    payload_no = input("Enter payload number [Y]: ").strip() or "1"
    if not audio_no.isdigit() or not payload_no.isdigit():
        raise ValueError("Audio number and payload number must be integers.")

    cover_path = build_cover_path(audio_no)
    share_input = input("Enter stegoaudio number (default=0): ").strip()
    share_no = int(share_input) if share_input else 0
    run_single(audio_no, payload_no, share_no, cover_path)


def run_nc_entropy_batch():
    from steganalysis import run_batch

    print("=" * 70)
    print("  NC AND ENTROPY ANALYSIS (batch + Excel)")
    print("=" * 70)

    audio_input = input("Total audio files (default=15): ").strip()
    payload_input = input("Total payloads (default=11): ").strip()
    shares_input = input("Total shares / n (blank = auto-detect): ").strip()
    run_batch(
        total_audio=int(audio_input) if audio_input else 15,
        total_payload=int(payload_input) if payload_input else 11,
        total_shares=int(shares_input) if shares_input else None,
    )


def run_evaluation():
    show_menu_evaluation()
    choice = input("Choose (1-5): ").strip()

    menu = EVALUATION_MENU.get(choice)
    if menu is None:
        print("Invalid choice. Please enter a number between 1 and 5.")
        return

    name, target = menu
    print("\nRunning {}...".format(name))

    if target == "detector_ml":
        run_detector_ml()
    elif target == "nc_entropy_single":
        run_nc_entropy_single()
    elif target == "nc_entropy_batch":
        run_nc_entropy_batch()
    else:
        run_modul(target)


def main():
    while True:
        show_menu()
        choice = input("Choose (1-4): ").strip()

        if choice == "4":
            print("Program Closed.")
            break

        menu = MENU.get(choice)
        if menu is None:
            print("Invalid choice. Please enter a number between 1 and 4.")
            continue

        name, name_modul = menu
        print("\nRunning {}...".format(name))

        try:
            if choice == "3":
                run_evaluation()
            elif choice == "1":
                run_embedding()
            else:
                run_modul(name_modul)
        except KeyboardInterrupt:
            print("\nCanceled by user.")
        except Exception as error:
            print("\nFailed to run {}: {}".format(name, error))


if __name__ == "__main__":
    main()
