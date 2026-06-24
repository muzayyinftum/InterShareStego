import random
from functools import reduce
from sympy import nextprime
import scipy.io.wavfile as scp
import numpy as np
import itertools
import math
import copy
import os
# embedding ============================================================================================================
def read_payload(file_payload):
    binary_data = list(open(file_payload))[0]
    binary_data = binary_data.split('\t')
    binary_data = [x.strip('ÿþ') for x in binary_data]

    binary_data = [x.strip('\x00') for x in binary_data]
    binary_data = ''.join(binary_data)
    return binary_data

def sampling(file_audio):
    rate, data = scp.read(file_audio)
    data = np.add(np.int16(data),[32768])
    return rate, data

def interpolation_linear(input_sampling):
    index_odd = [x for x in range (0, (len(input_sampling)*2) - 1) if x%2 == 1]
    index_even = [x for x in range (0, (len(input_sampling)*2)) if x%2 == 0]
    interpolated_sample = np.interp(index_odd, index_even, input_sampling)
    interpolated_sample = np.floor(interpolated_sample)
    return interpolated_sample

def determine_sample_space(interpolated_sample):
    bit = []
    for x in range(len(interpolated_sample)):
        if interpolated_sample[x] == 0:
            bit.append(0)
        else:
            bit.append(math.floor(math.sqrt(math.log(interpolated_sample[x],2))))
    return bit

def segmentation(payload, bit):
    index = 0
    processed_payload = []
    for x in bit:
        if index >= len(payload):
            break
        else:
            processed_payload.append(payload[index:index+x])
            index += x
    
    return processed_payload, len(processed_payload[-1])

def convert_bin_to_dec(payload):
    decimal = []
    isZeroInLast = False
    for x in range(len(payload)):
        decimal.append(int(payload[x],2))

    if(decimal[-1] == 0): # is the last decimal bit 0 ? because for the last index
        isZeroInLast = True
    return decimal, isZeroInLast

def get_prime_number(decimal_payload):
    max_value = max(decimal_payload)
    prime = nextprime(max_value)
    return prime

def shamir_secret_sharing(decimal_payload, prime, total_shares, min_shares):
    all_data_shares = []
    for x in range(len(decimal_payload)):
        data = split_secret(decimal_payload[x], prime, total_shares, min_shares)
        all_data_shares.append(data)
    
    return all_data_shares

def split_secret(secret, prime, total_shares, min_shares):

    coefficients = [secret]  # Coefficient first is the secret
    # Add k-1 random coefficients
    for _ in range(min_shares - 1):
        random_coefficient = random.randint(1, prime - 1)  # Random coefficient between 1 and prime-1
        coefficients.append(random_coefficient)
    
    shares = []
    for m in range(1, total_shares + 1):
        # evaluate the polynomial at x = m to get the share
        y = evaluate_polynomial(coefficients, m, prime)
        
        # add the share (x, y) to the list of shares
        shares.append(y)
    return shares

# Function for evaluating polynomial
def evaluate_polynomial(coefficients, x, prime):
    result = 0  # Initialize the initial result
    for i, c in enumerate(coefficients):
        term = (c * (x ** i)) % prime  # Calculate each term of the polynomial (c * x^i) mod prime
        result = (result + term) % prime  # Add the term to the result with modulus prime
    return result

def embedding(data, interpolated_sample, total_shares, last_bit, isZeroInLast):
    all_data = []
    for i in range(total_shares):
        single_audio = []
        for x in range(len(interpolated_sample)):
            if x <= len(data)-1:
                single_audio.append(interpolated_sample[x]-data[x][i])
            else:
                if x == len(interpolated_sample)-1:
                    single_audio.append(interpolated_sample[x] - i-1) #-1 karena index dimulai dari 0
                elif x == len(interpolated_sample)-2:
                    single_audio.append(interpolated_sample[x] - int(last_bit))
                elif x == len(interpolated_sample)-3 and isZeroInLast :
                    single_audio.append(interpolated_sample[x] - 1)
                else:
                    single_audio.append(interpolated_sample[x])
            
        all_data.append(single_audio)
    return all_data

def combine(embedded, original_sample):
    all_data = []
    for x in range(len(embedded)):
        single_stego = []
        index_stego = 0
        index_sample = 0
        index_embed = 0
        for y in range (0, len(original_sample)*2 - 1):
            if (index_stego % 2 == 0):
                single_stego.append(original_sample[index_sample])
                index_sample += 1
            else:
                single_stego.append(embedded[x][index_embed])
                index_embed += 1
            index_stego += 1
        all_data.append(single_stego)
    return all_data

def create_stego_audio(stego_data, filepath, cover_sample_rate):
    for x in range(len(stego_data)):
        new_filepath = filepath + str(x) + '.wav'
        process_1 = np.subtract(stego_data[x], [32768])
        stego_audio = np.array(process_1, dtype=np.int16)
        os.makedirs(os.path.dirname(new_filepath), exist_ok=True)
        L = len(stego_audio)
        n_cover = (L + 1) // 2
        stego_sample_rate = int(round(cover_sample_rate * L / n_cover))
        scp.write(new_filepath, stego_sample_rate, stego_audio)

# extracting ============================================================================================================
def extraction_sampling(file_stego_audio, stego_audio_no):
    all_stego_audio = []
    for x in stego_audio_no:
        new_filepath = file_stego_audio + str(x) + '.wav'
        rate, data = scp.read(new_filepath)
        data = np.add(np.int16(data),[32768])
        all_stego_audio.append(data)
    return rate, all_stego_audio

def separate(stego_sample):
    original_sample = []
    embedded = []
    for x in range(len(stego_sample)):
        single_original = []
        single_embedded = []
        for y in range(len(stego_sample[x])):
            if y % 2 == 0:
                single_original.append(stego_sample[x][y])
            else:
                single_embedded.append(stego_sample[x][y])
        original_sample.append(single_original)
        embedded.append(single_embedded)

    return original_sample, embedded

def extraction_interpolation_linear(embedded):
    all_interpolated = []
    for x in range(len(embedded)):
        index_odd = [x for x in range (0, (len(embedded[x])*2) - 1) if x%2 == 1]
        index_even = [x for x in range (0, (len(embedded[x])*2)) if x%2 == 0]
        interpolated_sample = np.interp(index_odd, index_even, embedded[x])
        interpolated_sample = np.floor(interpolated_sample)
        all_interpolated.append(interpolated_sample)
    return all_interpolated

#check apakah last index penyisipan antar semua min_shares sama
def check_last_index(embedded, interpolated_sample):
    last_index_of_embedding = []
    tmpZeroLast = []
    for x in range(len(embedded)):
        single_last_decimal = []
        single_zero_last = []
        for y in range(len(embedded[x])-1,-1,-1):
            value = int(interpolated_sample[x][y] - embedded[x][y])
            #jika value tidak sama dengan 0
            #jika y bukan di index terakhir (karena pada index terakhir untuk menyimpan no share)
            #jika y bukan di index kedua terakhir (karena pada index kedua terakhir untuk menyimpan last bit)
            #jika y bukan di index ketiga terakhir (karena pada index ini untuk menyimpan value (1/0) jika pada index terakhir adalah 0 )
            if value != 0 and y == len(embedded[x])-3:
                single_zero_last.append(value)
            if value != 0 and y != len(embedded[x])-1 and y != len(embedded[x])-2 and y != len(embedded[x])-3:
                single_last_decimal.append(y)
                break
        last_index_of_embedding.append(single_last_decimal)
        if len(single_zero_last) > 0:
            tmpZeroLast.append(single_zero_last)
    
    last_index = last_index_of_embedding[0][0]
    if len(last_index_of_embedding) > 1:
        value = all(sublist[0] == last_index for sublist in last_index_of_embedding)
        if value == False:
            flattened_list = [item for sublist in last_index_of_embedding for item in sublist]
            max_value = max(flattened_list)
            
            last_index = max_value
    else:
        if len(tmpZeroLast) > 0:
            last_index += 1
        value = True     
    return last_index, last_index_of_embedding

def check_next_prime(data_selisih):
    all_prime = []
    for x in range(len(data_selisih)):
        prime = get_prime_number(data_selisih[x])
        all_prime.append(prime)
    first_value = all_prime[0]
    value = all(x == first_value for x in all_prime)
    return value, first_value

def extraction_determine_selisih(embedded, interpolated_sample):
    last_index, all_last_index = check_last_index(embedded, interpolated_sample)
    
    data_selisih = []
    data_sahre_no = []
    last_bit = []
    for x in range(len(embedded)):
        single_decimal = []
        for y in range(len(embedded[x])):
            if y <= last_index:
                value = int(interpolated_sample[x][y] - embedded[x][y])
                single_decimal.append(value)
            if y == len(embedded[x])-2:
                single_last_bit = int(interpolated_sample[x][y] - embedded[x][y])
                last_bit.append(single_last_bit)
            if y == len(embedded[x])-1:
                single_share_no = int(interpolated_sample[x][y] - embedded[x][y])
                data_sahre_no.append(single_share_no)
        
        data_selisih.append(single_decimal)

    # is the prime value the same for all shares? if not, return False and the next prime value
    prime_value, next_prime = check_next_prime(data_selisih)

    if prime_value == False: #if not the same
        print("Nilai prime tidak sama :", next_prime)
        return

    return data_selisih, last_index+1, next_prime, data_sahre_no, last_bit[0]

def extraction_determine_sample_space(interpolated_sample, last_index):
    bit = []
    for x in range(len(interpolated_sample)):
        single_audio_bit = []
        for y in range(len(interpolated_sample[x])):
            if y == last_index:
                break
            else:
                single_audio_bit.append(math.floor(math.sqrt(math.log(interpolated_sample[x][y],2))))
        bit.append(single_audio_bit)
    return bit

def transpose_matrix(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def reconstruct_secret(data_selisih, prime_number, share_no):
    transposed_selisih = transpose_matrix(data_selisih)

    all_data = []
    for x in range(len(transposed_selisih)):
        combined_list = list(zip(share_no, transposed_selisih[x]))
        data = reconstruct_secret2(combined_list, prime_number)
        all_data.append(data)
    return all_data

def reconstruct_secret2(shares, prime):    
    def lagrange_interpolation(x, x_s, y_s, prime):
        total = 0
        for i in range(len(x_s)):
            xi, yi = x_s[i], y_s[i]
            prod = yi
            for j in range(len(x_s)):
                if i != j:
                    xj = x_s[j]
                    prod *= (x - xj) * pow(xi - xj, -1, prime)
                    prod %= prime
            total += prod
            total %= prime
        return total

    x_s, y_s = zip(*shares)
    return lagrange_interpolation(0, x_s, y_s, prime)

def check_all_nested_arrays_equal(nested_arr):
    def check_all_equal(arr):
        first_value = arr[0]
        for x in arr:
            if x != first_value:
                return False
        return True

    # Check if each sub-array has the same elements
    for sub_arr in nested_arr:
        if not check_all_equal(sub_arr):
            return False
    return True

def decimal_to_binary(decimal_payload, bit, last_bit):
    print(decimal_payload[-1])
    binary_payload = []
    new_bit = bit[0]
    for x in range(len(decimal_payload)):
        if x == len(decimal_payload)-1:
            binary_payload.append(np.binary_repr(decimal_payload[x], width=last_bit))
        else:
            binary_payload.append(np.binary_repr(decimal_payload[x], width=new_bit[x]))
    translated_payload = ''.join(binary_payload)
    # translated_payload = '\t'.join(translated_payload)

    # print(len(translated_payload))
    return translated_payload

def create_payload(byte_payload, filepath):
    os.makedirs(os.path.dirname(filepath),exist_ok=True)
    with open(filepath, 'w+') as file:
        file.write(byte_payload)
        file.close()  

def create_cover_audio(original_sample, filepath):
    unnormalize_data = np.subtract(original_sample,[32768])
    new_data_sample = np.array(unnormalize_data,dtype=np.int16)
    os.makedirs(os.path.dirname(filepath),exist_ok=True)
    scp.write(filepath, 44100, new_data_sample)


def calculate_ber(original_binary, extracted_binary):
    """
    Menghitung Bit Error Rate (BER) untuk mengukur tingkat keakuratan/security ekstraksi payload.
    
    BER = (Jumlah bit salah / Total bit) x 100%
    - BER 0% = payload diekstraksi sempurna (tingkat keamanan/reliability tinggi)
    - BER > 0% = ada bit yang salah (tingkat keamanan/reliability lebih rendah)
    
    Args:
        original_binary: string atau list biner payload asli (format: '01010101...' atau ['0','1',...])
        extracted_binary: string atau list biner payload hasil ekstraksi
    
    Returns:
        tuple: (ber_percentage, bit_errors, total_bits)
    """
    # Konversi ke string biner (handle list dari read_payload)
    def to_binary_str(data):
        if isinstance(data, (list, tuple)):
            return ''.join(str(b) for b in data if str(b) in '01')
        s = str(data).replace(' ', '').replace('\t', '')
        return ''.join(c for c in s if c in '01')
    
    original_binary = to_binary_str(original_binary)
    extracted_binary = to_binary_str(extracted_binary)
    
    # Ambil panjang minimum untuk perbandingan (hindari index out of range)
    min_length = min(len(original_binary), len(extracted_binary))
    
    if min_length == 0:
        return 100.0, 0, 0  # Tidak ada data untuk dibandingkan
    
    bit_errors = 0
    for i in range(min_length):
        if original_binary[i] != extracted_binary[i]:
            bit_errors += 1
    
    total_bits = min_length
    ber_percentage = (bit_errors / total_bits) * 100
    
    return ber_percentage, bit_errors, total_bits