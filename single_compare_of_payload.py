import scipy.io.wavfile
import numpy as np
import os

def read_payload(file_payload):
    with open(file_payload, mode='r') as file:
        data_payload = file.read()
    data_payload=[data_payload[x] for x in range (len(data_payload)) if data_payload[x]!='\t']
    data_payload=[x.strip('\x00') for x in data_payload]
    binary_data = '0b'+''.join(data_payload)
    binary_data = [x.strip('ÿþ') for x in binary_data]
    binary_data = binary_data[2::]
    if(binary_data[0]!='1' and binary_data[0]!='0') and (binary_data[1]!='1' and binary_data[1]!='0'):
        binary_data = binary_data[2::]

    return binary_data

def read_number_data(file):
    test = np.loadtxt(file,dtype=np.int)
    return test

def compare_data(data1, data2):
    hasil = 1
    if len(data1) != len(data2):
        print('Length of original = ', len(data1))
        print('Length of extracted = ', len(data2))

        return 0
    else:
        miss_data = 0
        for x in range (len(data1)):
            if data1[x] != data2[x]:
                print('Mismatch at index : ', x)
                print('original =', data1[x],'extracted =', data2[x])
                miss_data += 1
                hasil = 0
    return hasil

def create_payload(translated_payload, filepath):
    os.makedirs(os.path.dirname(filepath),exist_ok=True)
    with open(filepath, 'w') as file:
        file.write(translated_payload)
        file.close()
    return

def main():
    audio = '1'
    payload = '1'

    original_payload = 'DATASET/Payload/payload'+payload+'.txt'
    extract_payload = 'EXTRACTED/stego_audio'+audio+'_payload'+payload+'/payload.txt'

    data_ori_payload = read_payload(original_payload)
    data_ext_payload = read_payload(extract_payload)
    
    hasil_payload = compare_data(data_ori_payload, data_ext_payload)

    if hasil_payload == 0:
        print('Failed to extract correctly')
    else:
        print("Identical")

main()