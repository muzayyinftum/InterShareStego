import runpy


MENU = {
    "1": ("Embedding", "embedding"),
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


def tampilkan_menu():
    print("\n" + "=" * 70)
    print("  STEGOSHARE - Reversible Audio Steganography Using Secret Sharing")
    print("=" * 70)

    for pilihan in MENU_ORDER:
        nama, _ = MENU[pilihan]
        print("{}. {}".format(pilihan, nama))


def tampilkan_menu_evaluation():
    print("\n" + "=" * 70)
    print("  EVALUATION")
    print("=" * 70)

    for pilihan in EVALUATION_MENU_ORDER:
        nama, _ = EVALUATION_MENU[pilihan]
        print("{}. {}".format(pilihan, nama))


def jalankan_modul(nama_modul):
    runpy.run_module(nama_modul, run_name="__main__")


def jalankan_detector_ml():
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


def jalankan_nc_entropy_single():
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


def jalankan_nc_entropy_batch():
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


def jalankan_evaluation():
    tampilkan_menu_evaluation()
    pilihan = input("Choose (1-5): ").strip()

    menu = EVALUATION_MENU.get(pilihan)
    if menu is None:
        print("Invalid choice. Please enter a number between 1 and 5.")
        return

    nama, target = menu
    print("\nRunning {}...".format(nama))

    if target == "detector_ml":
        jalankan_detector_ml()
    elif target == "nc_entropy_single":
        jalankan_nc_entropy_single()
    elif target == "nc_entropy_batch":
        jalankan_nc_entropy_batch()
    else:
        jalankan_modul(target)


def main():
    while True:
        tampilkan_menu()
        pilihan = input("Choose (1-4): ").strip()

        if pilihan == "4":
            print("Program Closed.")
            break

        menu = MENU.get(pilihan)
        if menu is None:
            print("Invalid choice. Please enter a number between 1 and 4.")
            continue

        nama, nama_modul = menu
        print("\nRunning {}...".format(nama))

        try:
            if pilihan == "3":
                jalankan_evaluation()
            else:
                jalankan_modul(nama_modul)
        except KeyboardInterrupt:
            print("\nCanceled by user.")
        except Exception as error:
            print("\nFailed to run {}: {}".format(nama, error))


if __name__ == "__main__":
    main()
