import os
import time
import zipfile
from datetime import datetime

from methods import *


DEFAULT_STEGO_AUDIO_BASE = 'stego_audio/stego_audio1_payload1/stegoaudio'
DEFAULT_EXTRACTED_PAYLOAD = 'extracted/stego_audio1_payload1/payload.txt'
DEFAULT_EXTRACTED_AUDIO = 'extracted/stego_audio1_payload1/audio.wav'


def build_extraction_output_dir(zip_file, output_root='extracted/gui_extraction'):
    zip_name = os.path.splitext(os.path.basename(zip_file))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(output_root, '{}_{}'.format(zip_name, timestamp))


def extract_wav_files_from_zip(zip_file, output_dir):
    if not zipfile.is_zipfile(zip_file):
        raise ValueError("File input harus berupa ZIP.")

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


def run_single_extraction(zip_file, min_shares, output_dir=None):
    if output_dir is None:
        output_dir = build_extraction_output_dir(zip_file)

    start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    wav_files = extract_wav_files_from_zip(zip_file, output_dir)
    if len(wav_files) < min_shares:
        raise ValueError(
            "ZIP harus berisi minimal {} file stego-audio WAV. Ditemukan {} file.".format(
                min_shares,
                len(wav_files)
            )
        )

    selected_stego_files = wav_files[:min_shares]
    frame_rate, stego_sample = read_stego_samples(selected_stego_files)
    original_sample, embedded = separate(stego_sample)
    interpolated_sample = extraction_interpolation_linear(original_sample)

    extraction_result = extraction_determine_selisih(embedded, interpolated_sample)
    if extraction_result is None:
        raise ValueError("Proses ekstraksi gagal karena nilai prime pada stego-audio tidak konsisten.")

    data_selisih, last_index, prime_number, share_no, last_bit = extraction_result
    bit = extraction_determine_sample_space(interpolated_sample, last_index)
    decimal_payload = reconstruct_secret(data_selisih, prime_number, share_no)
    binary_payload = decimal_to_binary(decimal_payload, bit, last_bit)

    payload_output = os.path.join(output_dir, 'payload.txt')
    cover_audio_output = os.path.join(output_dir, 'audio.wav')
    create_payload(binary_payload, payload_output)
    create_cover_audio(original_sample[0], cover_audio_output)

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
    total_shares, min_shares = map(int, input("Masukkan dua angka sebagai (n, k): ").split())
    stego_audio_no = random.sample(range(total_shares), min_shares)

    frame_rate, stego_sample = extraction_sampling(DEFAULT_STEGO_AUDIO_BASE, stego_audio_no)
    original_sample, embedded = separate(stego_sample)
    interpolated_sample = extraction_interpolation_linear(original_sample)

    data_selisih, last_index, prime_number, share_no, last_bit = extraction_determine_selisih(embedded, interpolated_sample)
    bit = extraction_determine_sample_space(interpolated_sample, last_index)
    decimal_payload = reconstruct_secret(data_selisih, prime_number, share_no)
    binary_payload = decimal_to_binary(decimal_payload, bit, last_bit)

    create_payload(binary_payload, DEFAULT_EXTRACTED_PAYLOAD)
    create_cover_audio(original_sample[0], DEFAULT_EXTRACTED_AUDIO)


if __name__ == '__main__':
    main()
