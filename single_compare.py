"""Compare the original payload and audio with extracted results."""

import os

import numpy as np
import scipy.io.wavfile as wavfile


def read_payload(filepath):
    with open(filepath, mode='r') as file:
        data = file.read()

    data = [character for character in data if character != '\t']
    data = [character.strip('\x00') for character in data]
    binary_data = '0b' + ''.join(data)
    binary_data = [character.strip('\xff\xfe') for character in binary_data]
    binary_data = binary_data[2:]

    if (len(binary_data) >= 2 and
            binary_data[0] not in ('0', '1') and
            binary_data[1] not in ('0', '1')):
        binary_data = binary_data[2:]

    return binary_data


def sampling_audio(filepath):
    _, data = wavfile.read(filepath)
    data = np.asarray(data, dtype=np.int16)
    return np.add(data, [32768])


def compare_data(original_data, extracted_data, data_name):
    if len(original_data) != len(extracted_data):
        print("Original {} length : {}".format(data_name, len(original_data)))
        print("Extracted {} length: {}".format(data_name, len(extracted_data)))
        return False

    mismatch_indices = [
        index
        for index, (original, extracted) in enumerate(
            zip(original_data, extracted_data)
        )
        if not np.array_equal(original, extracted)
    ]

    if mismatch_indices:
        print("{} differences: {}".format(data_name.capitalize(), len(mismatch_indices)))
        print("First difference at index:", mismatch_indices[0])
        return False

    return True


def require_file(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError("File not found: {}".format(filepath))


def main():
    print("-" * 70)
    print("  Compare original payload and audio with extracted results")
    print("-" * 70)

    print("Select stego_audio[X]_payload[Y] to extract:")
    audio_no = input("Enter audio number [X] (default=1): ").strip() or '1'
    payload_no = input("Enter payload number [Y] (default=1): ").strip() or '1'

    if not audio_no.isdigit() or not payload_no.isdigit():
        raise ValueError("Audio and payload numbers must be numeric.")

    original_payload = 'DATASET/Payload/payload{}.txt'.format(payload_no)
    extracted_payload = 'EXTRACTED/stego_audio{}_payload{}/payload.txt'.format(
        audio_no, payload_no
    )
    original_audio = 'DATASET/Audio/data{}_mono.wav'.format(audio_no)
    extracted_audio = 'EXTRACTED/stego_audio{}_payload{}/audio.wav'.format(
        audio_no, payload_no
    )

    for filepath in (
        original_payload, extracted_payload, original_audio, extracted_audio
    ):
        require_file(filepath)

    payload_equal = compare_data(
        read_payload(original_payload),
        read_payload(extracted_payload),
        'payload'
    )
    audio_equal = compare_data(
        sampling_audio(original_audio),
        sampling_audio(extracted_audio),
        'audio'
    )

    print("\nComparison results:")
    print("Payload:", "Exact" if payload_equal else "Different")
    print("Audio  :", "Exact" if audio_equal else "Different")


if __name__ == '__main__':
    main()
