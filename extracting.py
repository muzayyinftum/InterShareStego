import os
import time
import zipfile
from datetime import datetime

from methods import *


def build_extraction_output_dir(zip_file, output_root='EXTRACTED'):
    zip_name = os.path.splitext(os.path.basename(zip_file))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(output_root, '{}_{}'.format(zip_name, timestamp))


def extract_wav_files_from_zip(zip_file, output_dir):
    if not zipfile.is_zipfile(zip_file):
        raise ValueError("File input must be a ZIP file.")

    input_dir = os.path.join(output_dir, 'input_stego_audio')
    os.makedirs(input_dir, exist_ok=True)

    wav_files = []
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for member in zip_ref.infolist():
            if member.is_dir() or not member.filename.lower().endswith('.wav'):
                continue

            filename = os.path.basename(member.filename)
            if not filename:
                continue

            target_path = os.path.join(input_dir, filename)
            with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                target.write(source.read())
            wav_files.append(target_path)

    wav_files.sort()
    return wav_files


def read_stego_samples(stego_files):
    all_stego_audio = []
    sample_rate = None

    for stego_file in stego_files:
        rate, data = scp.read(stego_file)
        if sample_rate is None:
            sample_rate = rate

        data = np.add(np.int16(data), [32768])
        all_stego_audio.append(data)

    return sample_rate, all_stego_audio


def extract_payload_and_audio(stego_sample, output_dir):
    original_sample, embedded = separate(stego_sample)
    interpolated_sample = extraction_interpolation_linear(original_sample)

    extraction_result = extraction_difference_determination(embedded, interpolated_sample)
    if extraction_result is None:
        raise ValueError("Extraction failed because the prime values in the stego-audio were inconsistent.")

    difference_data, last_index, prime_number, share_no, last_bit = extraction_result
    bit = extraction_determine_sample_space(interpolated_sample, last_index)
    decimal_payload = reconstruct_secret(difference_data, prime_number, share_no)
    binary_payload = decimal_to_binary(decimal_payload, bit, last_bit)

    payload_output = os.path.join(output_dir, 'payload.txt')
    cover_audio_output = os.path.join(output_dir, 'audio.wav')
    create_payload(binary_payload, payload_output)
    create_cover_audio(original_sample[0], cover_audio_output)

    return payload_output, cover_audio_output


def run_single_extraction(zip_file, min_shares, output_dir=None):
    if output_dir is None:
        output_dir = build_extraction_output_dir(zip_file)

    start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    wav_files = extract_wav_files_from_zip(zip_file, output_dir)
    if len(wav_files) < min_shares:
        raise ValueError(
            "ZIP file must contain at least {} WAV stego-audio files. Found {} files.".format(
                min_shares,
                len(wav_files)
            )
        )

    selected_stego_files = wav_files[:min_shares]
    frame_rate, stego_sample = read_stego_samples(selected_stego_files)
    payload_output, cover_audio_output = extract_payload_and_audio(stego_sample, output_dir)

    return {
        'output_dir': output_dir,
        'payload_file': payload_output,
        'cover_audio_file': cover_audio_output,
        'selected_stego_files': selected_stego_files,
        'zip_wav_count': len(wav_files),
        'min_shares': min_shares,
        'runtime': time.time() - start,
    }


def main():
    print("Select stego_audio[X]_payload[Y] to extract:")

    audio_no = input("Enter audio number [X]: ").strip()
    payload_no = input("Enter payload number [Y]: ").strip()

    if not audio_no.isdigit() or not payload_no.isdigit():
        raise ValueError("Audio number and payload number must be integers.")

    stego_audio_base = 'STEGOAUDIO/stego_audio{}_payload{}/stegoaudio'.format(
        audio_no,
        payload_no
    )
    output_dir = 'EXTRACTED/stego_audio{}_payload{}'.format(audio_no, payload_no)

    total_shares, min_shares = map(
        int,
        input("Enter total and minimum shares (n k): ").split()
    )
    if total_shares < 1 or not 1 <= min_shares <= total_shares:
        raise ValueError("Value of shares must satisfy 1 <= k <= n.")

    missing_files = [
        '{}{}.wav'.format(stego_audio_base, index)
        for index in range(total_shares)
        if not os.path.isfile('{}{}.wav'.format(stego_audio_base, index))
    ]
    if missing_files:
        raise FileNotFoundError(
            "Incomplete stego-audio file. First file not found: {}".format(
                missing_files[0]
            )
        )

    stego_audio_no = random.sample(range(total_shares), min_shares)

    frame_rate, stego_sample = extraction_sampling(stego_audio_base, stego_audio_no)
    payload_output, cover_audio_output = extract_payload_and_audio(stego_sample, output_dir)

    print("\nExtraction completed successfully.")
    print("Share used:", stego_audio_no)
    print("Extracted payload:", payload_output)
    print("Extracted audio:", cover_audio_output)


if __name__ == '__main__':
    main()
