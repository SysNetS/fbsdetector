#!/usr/bin/env python
# coding: utf-8
import os
import csv
import requests
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def align_lists(list1, list2, list2_values):
    list2_values_aligned = []
    list2_aligned = []
    for x in list1:
        if x not in list2:
            list2_values_aligned.append('')
        else:
            idx = list2.index(x)
            list2_values_aligned.append(list2_values[idx])
    return list2_values_aligned


# Get Packet Fields

def get_packet_field_names(packet):
    packet_fields = []
    nested_field_no = 0
    for proto in packet.findall('proto'):
        if proto.get('name') == 'nas-eps':
            for field in proto.findall('field'):
                field_name = field.get('name').replace(".", "_")
                if field_name == '':
                    continue
                packet_fields.extend([field_name + '_show', field_name + '_value', field_name + '_size'])
    return packet_fields


# Make Dataframe from Packet Features

def makeDataframe(xmlfile):
    dataframe_list = []
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    packets = root.findall('packet')
    df = pd.DataFrame()
    i = 0
    
    column_names = []
    values = []

    column_names = []
    for packet in packets:
        columns = get_packet_field_names(packet)
        column_names = set(column_names).union(set(columns))
    column_names = list(column_names)
    
    for packet in packets:
        packet_fields = []
        for proto in packet.findall('proto'):
            selected_fields = []
            if proto.get('name') == 'nas-eps':
                for field in proto.findall('field'):
                    field_name = field.get('name').replace(".", "_")
                    if field_name == '':
                        continue
#                     if field_name == 'nas_eps_nas_msg_emm_type':
#                         print(field.get('value'))
                    packet_fields.extend([field.get('show'), field.get('value'), field.get('size')])
        columns = get_packet_field_names(packet)
        values.append(align_lists(column_names, columns, packet_fields))
    return pd.DataFrame(values, columns=column_names)


# Prepare Dataframe from PCAP File

def prepare_dataframe_from_pcap_file(input_file, output_file):
    xml_output_file = input_file.replace("pcap", "xml")
    os.system("tshark -r " + input_file + " -T pdml > " + xml_output_file)

    df = makeDataframe(xml_output_file)

    df.to_csv(output_file)

    df = pd.read_csv(output_file, on_bad_lines='skip')
    
    df.to_csv(output_file)


# Label Data
# Energy Depletion Attack
# Exp  1 - 10

input_file = "../dataset/multi-step-attacks/energy-depletion-attack/exp-1/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/energy-depletion-attack/exp-1/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_eda_01 = pd.read_csv(output_file)

labels_eda_01 = []
labels_eda_01 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 
                 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

df_eda_01['label'] = labels_eda_01
df_eda_01['label'].value_counts()

input_file = "../dataset/multi-step-attacks/energy-depletion-attack/exp-2/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/energy-depletion-attack/exp-2/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_eda_02 = pd.read_csv(output_file)

labels_eda_02 = []

labels_eda_02 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0]

df_eda_02['label'] = labels_eda_02
df_eda_02['label'].value_counts()

input_file = "../dataset/multi-step-attacks/energy-depletion-attack/exp-3/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/energy-depletion-attack/exp-3/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_eda_03 = pd.read_csv(output_file)

labels_eda_03 = []

labels_eda_03 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 0]

df_eda_03['label'] = labels_eda_03
df_eda_03['label'].value_counts()

input_file = "../dataset/multi-step-attacks/energy-depletion-attack/exp-4/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/energy-depletion-attack/exp-4/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_eda_04 = pd.read_csv(output_file)

labels_eda_04 = []

labels_eda_04 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 0]

df_eda_04['label'] = labels_eda_04
df_eda_04['label'].value_counts()

input_file = "../dataset/multi-step-attacks/energy-depletion-attack/exp-5/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/energy-depletion-attack/exp-5/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_eda_05 = pd.read_csv(output_file)

labels_eda_05 = []

labels_eda_05 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 0]

df_eda_05['label'] = labels_eda_05
df_eda_05['label'].value_counts()

input_file = "../dataset/multi-step-attacks/energy-depletion-attack/exp-6/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/energy-depletion-attack/exp-6/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_eda_06 = pd.read_csv(output_file)

labels_eda_06 = []

labels_eda_06 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 
                 
                 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 0]

df_eda_06['label'] = labels_eda_06
df_eda_06['label'].value_counts()

input_file = "../dataset/multi-step-attacks/energy-depletion-attack/exp-7/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/energy-depletion-attack/exp-7/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_eda_07 = pd.read_csv(output_file)

labels_eda_07 = []

labels_eda_07 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0]

df_eda_07['label'] = labels_eda_07
df_eda_07['label'].value_counts()

input_file = "../dataset/multi-step-attacks/energy-depletion-attack/exp-8/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/energy-depletion-attack/exp-8/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_eda_08 = pd.read_csv(output_file)

labels_eda_08 = []

labels_eda_08 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 
                 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0]

df_eda_08['label'] = labels_eda_08
df_eda_08['label'].value_counts()

input_file = "../dataset/multi-step-attacks/energy-depletion-attack/exp-9/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/energy-depletion-attack/exp-9/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_eda_09 = pd.read_csv(output_file)

labels_eda_09 = []

labels_eda_09 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 0]

df_eda_09['label'] = labels_eda_09
df_eda_09['label'].value_counts()

input_file = "../dataset/multi-step-attacks/energy-depletion-attack/exp-10/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/energy-depletion-attack/exp-10/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_eda_10 = pd.read_csv(output_file)

labels_eda_10 = []

labels_eda_10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                 0, 0, 0, 0, 0]

df_eda_10['label'] = labels_eda_10
df_eda_10['label'].value_counts()

# nas-counter-desync-attack
# Exp 1 - 6
input_file = "../dataset/multi-step-attacks/nas-counter-desync-attack/exp-1/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/nas-counter-desync-attack/exp-1/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_ncda_01 = pd.read_csv(output_file)

labels_ncda_01 = []

for i in range(df_ncda_01.shape[0]):
    if i >= 10 and i < 1545 :
        labels_ncda_01.append(2)
    else:
        labels_ncda_01.append(0)

df_ncda_01['label'] = labels_ncda_01
df_ncda_01['label'].value_counts()

input_file = "../dataset/multi-step-attacks/nas-counter-desync-attack/exp-2/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/nas-counter-desync-attack/exp-2/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_ncda_02 = pd.read_csv(output_file)

labels_ncda_02 = []

for i in range(df_ncda_02.shape[0]):
    if i > 10 and i < 1545 :
        labels_ncda_02.append(2)
    else:
        labels_ncda_02.append(0)

df_ncda_02['label'] = labels_ncda_02
df_ncda_02['label'].value_counts()

input_file = "../dataset/multi-step-attacks/nas-counter-desync-attack/exp-3/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/nas-counter-desync-attack/exp-3/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_ncda_03 = pd.read_csv(output_file)

labels_ncda_03 = []

for i in range(df_ncda_03.shape[0]):
    if i > 10 and i < 6150 :
        labels_ncda_03.append(2)
    else:
        labels_ncda_03.append(0)

df_ncda_03['label'] = labels_ncda_03
df_ncda_03['label'].value_counts()

input_file = "../dataset/multi-step-attacks/nas-counter-desync-attack/exp-4/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/nas-counter-desync-attack/exp-4/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_ncda_04 = pd.read_csv(output_file)

labels_ncda_04 = []

for i in range(df_ncda_04.shape[0]):
    if i > 10 and i < 1545 :
        labels_ncda_04.append(2)
    else:
        labels_ncda_04.append(0)

df_ncda_04['label'] = labels_ncda_04
df_ncda_04['label'].value_counts()

input_file = "../dataset/multi-step-attacks/nas-counter-desync-attack/exp-5/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/nas-counter-desync-attack/exp-5/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_ncda_05 = pd.read_csv(output_file)

labels_ncda_05 = []

for i in range(df_ncda_05.shape[0]):
    if i > 10 and i < 1545 :
        labels_ncda_05.append(2)
    else:
        labels_ncda_05.append(0)
        
df_ncda_05['label'] = labels_ncda_05
df_ncda_05['label'].value_counts()

input_file = "../dataset/multi-step-attacks/nas-counter-desync-attack/exp-6/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/nas-counter-desync-attack/exp-6/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_ncda_06 = pd.read_csv(output_file)

labels_ncda_06 = []

for i in range(df_ncda_06.shape[0]):
    if i > 10:
        labels_ncda_06.append(2)
    else:
        labels_ncda_06.append(0)
        
df_ncda_06['label'] = labels_ncda_06
df_ncda_06['label'].value_counts()


# numb-attack
input_file = "../dataset/multi-step-attacks/numb-attack/exp-2/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/numb-attack/exp-2/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_nmba_01 = pd.read_csv(output_file)

labels_nmba_01 = []

labels_nmba_01 = [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 
                  3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 
                  0, 0, 0, 0]

df_nmba_01['label'] = labels_nmba_01

input_file = "../dataset/multi-step-attacks/numb-attack/exp-3/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/numb-attack/exp-3/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_nmba_02 = pd.read_csv(output_file)

labels_nmba_02 = []

labels_nmba_02 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
                  3, 3, 3, 0, 0, 0, 0, 0, 0, 0,
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  3, 3, 3, 3, 3, 3, 3, 3, 0, 0,
                  0, 0, 0, 0, 0, 0]

df_nmba_02['label'] = labels_nmba_02

input_file = "../dataset/multi-step-attacks/numb-attack/exp-4/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/numb-attack/exp-4/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_nmba_03 = pd.read_csv(output_file)

labels_nmba_03 = []

labels_nmba_03 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 3, 3, 3,
                  3, 3, 3, 3, 3, 0, 0, 0, 0, 0,
                  0, 0, 0]

df_nmba_03['label'] = labels_nmba_03

input_file = "../dataset/multi-step-attacks/numb-attack/exp-5/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/numb-attack/exp-5/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_nmba_04 = pd.read_csv(output_file)

labels_nmba_04 = []

labels_nmba_04 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 
                  0, 0, 0, 0, 0, 0]

df_nmba_04['label'] = labels_nmba_04

input_file = "../dataset/multi-step-attacks/numb-attack/exp-6/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/numb-attack/exp-6/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_nmba_05 = pd.read_csv(output_file)

labels_nmba_05 = []

labels_nmba_05 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 
                  0, 0, 0, 0, 0, 0]

df_nmba_05['label'] = labels_nmba_05

input_file = "../dataset/multi-step-attacks/numb-attack/exp-7/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/numb-attack/exp-7/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_nmba_06 = pd.read_csv(output_file)

labels_nmba_06 = []

labels_nmba_06 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 
                  0]

df_nmba_06['label'] = labels_nmba_06

# paging-channel-hijacking-attack
input_file = "../dataset/multi-step-attacks/paging-channel-hijacking-attack/exp-1/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/paging-channel-hijacking-attack/exp-1/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_pcha_01 = pd.read_csv(output_file)

labels_pcha_01 = []

labels_pcha_01 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
                  4, 0]

df_pcha_01['label'] = labels_pcha_01

input_file = "../dataset/multi-step-attacks/paging-channel-hijacking-attack/exp-2/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/paging-channel-hijacking-attack/exp-2/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_pcha_02 = pd.read_csv(output_file)

labels_pcha_02 = []

labels_pcha_02 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
                  0, 0, 0, 0, 0, 0, 0, 0, 0]

df_pcha_02['label'] = labels_pcha_02

input_file = "../dataset/multi-step-attacks/paging-channel-hijacking-attack/exp-3/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/paging-channel-hijacking-attack/exp-3/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_pcha_03 = pd.read_csv(output_file)

labels_pcha_03 = []

labels_pcha_03 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
                  4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 
                  0, 0, 0, 0]

df_pcha_03['label'] = labels_pcha_03

input_file = "../dataset/multi-step-attacks/paging-channel-hijacking-attack/exp-4/ue/ue_nas.pcap"
output_file = "../dataset/multi-step-attacks/paging-channel-hijacking-attack/exp-4/ue/ue_nas.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df_pcha_04 = pd.read_csv(output_file)

labels_pcha_04 = []

labels_pcha_04 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
                  4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 
                  0, 0, 0, 0]

df_pcha_04['label'] = labels_pcha_04

# Benign Data
input_file = "../dataset/fake_exp32.pcap"
output_file = "../dataset/fake_exp32.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df32 = pd.read_csv(output_file)

labels32 = [0] * df32.shape[0]

df32['label'] = labels32
df32['label'].value_counts()

input_file = "../dataset/fake_exp31.pcap"
output_file = "../dataset/fake_exp31.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df31 = pd.read_csv(output_file)

labels31 = [0] * df31.shape[0]

df31['label'] = labels31
df31['label'].value_counts()

input_file = "../dataset/fake_exp24.pcap"
output_file = "../dataset/fake_exp24.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df24 = pd.read_csv(output_file)

labels24 = [0] * df24.shape[0]

df24['label'] = labels24
df24['label'].value_counts()

input_file = "../dataset/fake_exp21.pcap"
output_file = "../dataset/fake_exp21.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df21 = pd.read_csv(output_file)

labels21 = [0] * df21.shape[0]

df21['label'] = labels21
df21['label'].value_counts()

input_file = "../dataset/fake_exp20.pcap"
output_file = "../dataset/fake_exp20.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df20 = pd.read_csv(output_file)

labels20 = [0] * df20.shape[0]

df20['label'] = labels20
df20['label'].value_counts()

input_file = "../dataset/fake_exp19.pcap"
output_file = "../dataset/fake_exp19.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df19 = pd.read_csv(output_file)

labels19 = [0] * df19.shape[0]

df19['label'] = labels19
df19['label'].value_counts()

input_file = "../dataset/fake_exp17.pcap"
output_file = "../dataset/fake_exp17.csv"

prepare_dataframe_from_pcap_file(input_file, output_file)
df17 = pd.read_csv(output_file)

labels17 = [0] * df17.shape[0]

df17['label'] = labels17
df17['label'].value_counts()

# Append Data
# df_eda = df_eda_01.append(df_eda_02).append(df_eda_03).append(df_eda_04).append(df_eda_05).append(df_eda_06).append(df_eda_07).append(df_eda_08).append(df_eda_09).append(df_eda_10)
# df_ncda = df_ncda_01 #.append(df_ncda_02).append(df_ncda_03).append(df_ncda_04).append(df_ncda_05).append(df_ncda_06)
# df_nmba = df_nmba_01.append(df_nmba_02).append(df_nmba_03).append(df_nmba_04).append(df_nmba_05).append(df_nmba_06)
# df_pcha = df_pcha_01.append(df_pcha_02).append(df_pcha_03).append(df_pcha_04)
# df_benign = df17.append(df19).append(df20).append(df21).append(df24).append(df31).append(df32)

import pandas as pd

df_eda = pd.concat([df_eda_01, df_eda_02, df_eda_03, df_eda_04, df_eda_05,
                    df_eda_06, df_eda_07, df_eda_08, df_eda_09, df_eda_10], ignore_index=True)
# df_ncda = pd.concat([df_ncda_01, df_ncda_02, df_ncda_03, df_ncda_04, df_ncda_05, df_ncda_06], ignore_index=True)
df_ncda = df_ncda_01
df_nmba = pd.concat([df_nmba_01, df_nmba_02, df_nmba_03, df_nmba_04, df_nmba_05, df_nmba_06], ignore_index=True)
df_pcha = pd.concat([df_pcha_01, df_pcha_02, df_pcha_03, df_pcha_04], ignore_index=True)
df_benign = pd.concat([df17, df19, df20, df21, df24, df31, df32], ignore_index=True)
df_ncda = df_ncda.loc[:500]

df_eda['label'].value_counts()
df_ncda['label'].value_counts()
df_nmba['label'].value_counts()
df_pcha['label'].value_counts()
df_eda.shape, df_ncda.shape, df_nmba.shape, df_pcha.shape

# df = df_eda.append(df_nmba).append(df_pcha).append(df_ncda).append(df_benign)
df = pd.concat([df_eda, df_nmba, df_pcha, df_ncda, df_benign], ignore_index=True)
df['label'].value_counts()
df.shape
df['nas-eps_nas_msg_emm_type_value'].value_counts()
df[df['nas-eps_nas_msg_emm_type_value']=='48']['label'].value_counts()

df0 = df[(df['nas-eps_nas_msg_emm_type_value'] == '48') & (df['label'] == 0)]
df1 = df[(df['nas-eps_nas_msg_emm_type_value'] == '48') & (df['label'] == 1)]
df2 = df[(df['nas-eps_nas_msg_emm_type_value'] == '48') & (df['label'] == 2)]
df3 = df[(df['nas-eps_nas_msg_emm_type_value'] == '48') & (df['label'] == 3)]
df4 = df[(df['nas-eps_nas_msg_emm_type_value'] == '48') & (df['label'] == 4)]

def compare_rows_between_dataframes(df1, df2, row_idx1, row_idx2):
    row1 = df1.iloc[row_idx1]
    row2 = df2.iloc[row_idx2]
    
    differing_columns = []
    for column in df1.columns:
        value1 = row1[column]
        value2 = row2[column]

        # Check for NaN values
        if pd.isna(value1) and pd.isna(value2):
            continue  # Skip if both values are NaN
        elif pd.isna(value1) or pd.isna(value2):
            differing_columns.append(column)
        elif value1 != value2:
            differing_columns.append(column)
    
    if differing_columns:
        print(f"Differences between row {row_idx1} of DataFrame 1 and row {row_idx2} of DataFrame 2:")
        for column in differing_columns:
            print(f"{column}: {row1[column]} -> {row2[column]}")
    else:
        print(f"Rows {row_idx1} of DataFrame 1 and {row_idx2} of DataFrame 2 are identical.")

compare_rows_between_dataframes(df2, df1, 0, 0)
df.to_csv("../dataset/multi_step_attack_data.csv")

# df_eda_01.to_csv("../dataset/multi-step-attacks/processed/eda_01.csv")
# df_eda_02.to_csv("../dataset/multi-step-attacks/processed/eda_02.csv")
# df_eda_03.to_csv("../dataset/multi-step-attacks/processed/eda_03.csv")
# df_eda_04.to_csv("../dataset/multi-step-attacks/processed/eda_04.csv")
# df_eda_05.to_csv("../dataset/multi-step-attacks/processed/eda_05.csv")
# df_eda_06.to_csv("../dataset/multi-step-attacks/processed/eda_06.csv")
# df_eda_07.to_csv("../dataset/multi-step-attacks/processed/eda_07.csv")
# df_eda_08.to_csv("../dataset/multi-step-attacks/processed/eda_08.csv")
# df_eda_09.to_csv("../dataset/multi-step-attacks/processed/eda_09.csv")
# df_eda_10.to_csv("../dataset/multi-step-attacks/processed/eda_10.csv")


# df_nmba_01.to_csv("../dataset/multi-step-attacks/processed/nmba_01.csv")
# df_nmba_02.to_csv("../dataset/multi-step-attacks/processed/nmba_02.csv")
# df_nmba_03.to_csv("../dataset/multi-step-attacks/processed/nmba_03.csv")
# df_nmba_04.to_csv("../dataset/multi-step-attacks/processed/nmba_04.csv")
# df_nmba_05.to_csv("../dataset/multi-step-attacks/processed/nmba_05.csv")
# df_nmba_06.to_csv("../dataset/multi-step-attacks/processed/nmba_06.csv")


# df_pcha_01.to_csv("../dataset/multi-step-attacks/processed/pcha_01.csv")
# df_pcha_02.to_csv("../dataset/multi-step-attacks/processed/pcha_02.csv")
# df_pcha_03.to_csv("../dataset/multi-step-attacks/processed/pcha_03.csv")
# df_pcha_04.to_csv("../dataset/multi-step-attacks/processed/pcha_04.csv")


# df_ncda_01.to_csv("../dataset/multi-step-attacks/processed/ncda_01.csv")
# df_ncda_02.to_csv("../dataset/multi-step-attacks/processed/ncda_02.csv")
# df_ncda_03.to_csv("../dataset/multi-step-attacks/processed/ncda_03.csv")
# df_ncda_04.to_csv("../dataset/multi-step-attacks/processed/ncda_04.csv")
# df_ncda_05.to_csv("../dataset/multi-step-attacks/processed/ncda_05.csv")
# df_ncda_06.to_csv("../dataset/multi-step-attacks/processed/ncda_06.csv")