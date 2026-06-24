import runpy


MENU = {
    "1": ("Embedding", "embedding"),
    "2": ("Extracting", "extracting"),
    "3": ("Compare payload and audio", "single_compare"),
    "4": ("Check audio quality", "quality_evaluation"),
    "5": ("Steganalysis", "steganalysis"),
}


def tampilkan_menu():
    print("\n" + "=" * 70)
    print("  STEGOSHARE - Reversible Audio Steganography Using Secret Sharing")
    print("=" * 70)

    for pilihan, (nama, _) in MENU.items():
        print("{}. {}".format(pilihan, nama))
    print("0. Exit")


def jalankan_modul(nama_modul):
    runpy.run_module(nama_modul, run_name="__main__")


def main():
    while True:
        tampilkan_menu()
        pilihan = input("Choose (0-5): ").strip()

        if pilihan == "0":
            print("Program Closed.")
            break

        menu = MENU.get(pilihan)
        if menu is None:
            print("Invalid choice. Please enter a number between 0 and 5.")
            continue

        nama, nama_modul = menu
        print("\nRunning {}...".format(nama))

        try:
            jalankan_modul(nama_modul)
        except KeyboardInterrupt:
            print("\nCanceled by user.")
        except Exception as error:
            print("\nFailed to run {}: {}".format(nama, error))


if __name__ == "__main__":
    main()
