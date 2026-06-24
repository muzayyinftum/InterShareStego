import runpy


MENU = {
    "1": ("Embedding", "embedding"),
    "2": ("Extracting", "extracting"),
    "3": ("Compare payload", "single_compare_of_payload"),
    "4": ("Compare audio", "single_compare_of_audio"),
    "5": ("Check audio quality", "single_check_of_quality"),
    "6": ("Steganalysis detector", "steganalysis_detector"),
    "7": ("Advanced steganalysis", "steganalysis2"),
}


def tampilkan_menu():
    print("\n=== StegoShare ===")
    for pilihan, (nama, _) in MENU.items():
        print("{}. {}".format(pilihan, nama))
    print("0. Exit")


def jalankan_modul(nama_modul):
    runpy.run_module(nama_modul, run_name="__main__")


def main():
    while True:
        tampilkan_menu()
        pilihan = input("Choose (0-7): ").strip()

        if pilihan == "0":
            print("Program Closed.")
            break

        menu = MENU.get(pilihan)
        if menu is None:
            print("Invalid choice. Please enter a number between 0 and 7.")
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
