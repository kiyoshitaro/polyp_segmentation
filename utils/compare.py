
from matplotlib import pyplot as plt
min_val = 850
path_colors = [
    # ("logs/GCPA_ASPP_orgdataset/test_GCPAASPPNet_2021-03-12 06:53:41.989173_5_Kvasir.log",'r',0),
    # ("logs/GCPA_PSP_orgdataset/test_GCPAPSPNet_2021-03-12 23:10:22.569401_5_Kvasir.log",'lime',0),

    ("logs/GCPA_GALDv9_orgdataset/test_GCPAGALDNetv9_2021-03-10 17:37:26.138396_5_Kvasir.log",'g',0),
    ("logs/GCPA_GALDv8_orgdataset/test_GCPAGALDNetv8_2021-03-10 10:58:50.415639_5_Kvasir.log",'b',0),
    # ("logs/GCPA_GALDv7_orgdataset/test_2021-03-10 05:38:19.526538_GCPAGALDNetv7_5_Kvasir.log",'m',0),
    ("logs/GCPA_orgdataset/test_2021-03-08 22:14:28.523723_GCPANet_5.log",'orange',1),

    # ("logs/PraNetvFAM2GALD_orgdataset/test_PraNetvFAM2GALD_2021-03-11 07:50:40.020118_5_Kvasir.log",'cyan',0),
    ("logs/GCPA_GALDv6_orgdataset/test_2021-03-10 01:15:09.192406_GCPAGALDNetv6_5_Kvasir.log",'y',1),
    # ("logs/GCPA_GALDv4_orgdataset/test_2021-03-07 23:18:34.142658_GCPAGALDNetv4_5.log",'black',1),
    # ("logs/pranetGALD_orgdataset/test_2021-03-06 15:21:49.319834_PraNetvGALD_3.log",'brown',1)
    # ("logs/GCPA_GALDv5_orgdataset/test_2021-03-09 15:07:49.599382_GCPAGALDNetv5_5_Kvasir.log",'gray',1),
    # ("logs/PraNetvFAMGALD_orgdataset/test_PraNetvFAMGALD_2021-03-11 07:29:18.560009_5_Kvasir.log",'white',0),
]

path_colors = [
    # ("logs/GCPA_GALDv5_orgdataset/test_2021-03-09 15:11:29.240565_GCPAGALDNetv5_5_CVC-ClinicDB.log",'gray',1),
    ("logs/GCPA_GALDv8_orgdataset/test_GCPAGALDNetv8_2021-03-10 11:02:34.064225_5_CVC-ClinicDB.log",'b',0),
    ("logs/GCPA_PSP_orgdataset/test_GCPAPSPNet_2021-03-12 23:10:44.953094_5_CVC-ClinicDB.log",'lime',0),
    # ("logs/GCPA_GALDv6_orgdataset/test_2021-03-10 01:15:52.975074_GCPAGALDNetv6_5_CVC-ClinicDB.log",'y',1),
    # ("logs/GCPA_GALDv7_orgdataset/test_2021-03-10 05:40:58.714650_GCPAGALDNetv7_5_CVC-ClinicDB.log",'m',0),
    ("logs/GCPA_ASPP_orgdataset/test_GCPAASPPNet_2021-03-12 06:57:14.967380_5_CVC-ClinicDB.log",'r',0),
    # ("logs/GCPA_GALDv9_orgdataset/test_GCPAGALDNetv9_2021-03-10 17:41:31.315643_5_CVC-ClinicDB.log",'g',0),
    # ("logs/GCPA_GALDv4_orgdataset/test_2021-03-08 00:19:23.764247_GCPAGALDNetv4_5.log",'black',1),
    # ("logs/pranetGALD_orgdataset/test_2021-03-06 15:29:40.884150_PraNetvGALD_3.log",'brown',1),
    # ("logs/PraNetvFAM2GALD_orgdataset/test_PraNetvFAM2GALD_2021-03-11 08:10:47.353152_5_CVC-ClinicDB.log",'cyan',0),
    # ("logs/PraNetvFAMGALD_orgdataset/test_PraNetvFAMGALD_2021-03-11 07:29:58.243561_5_CVC-ClinicDB.log",'white',0),
]
for path, color, ty in path_colors:
    data = (open(path,"r").read().split("\n"))
    if ty == 0:
        x = [int(data[i][-3:]) for i in range(2, len(data), 6) if len(data[i][-3:]) > 1]
        y = [(int(data[i][-3:]) - min_val) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3]
        # y = [(int(data[i][-21:-18]) - min_val) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3]
    elif ty == 1:
        x = [int(data[i][-3:]) for i in range(2, len(data), 5) if len(data[i][-3:]) > 1]
        y = [(int(data[i][-3:]) - min_val) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3]
        # y = [(int(data[i][-21:-18]) - min_val) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3]
    plt.bar(x, y, color=color)

path_colors = [
    # ("logs/_GCPA_ASPP_orgdataset/test_GCPAASPPNet_2021-03-13 12:34:26.008870_5_Kvasir.log",'r',0),
    # ("logs/_GCPA_PSP_orgdataset/test_GCPAPSPNet_2021-03-13 11:54:56.156760_5_Kvasir.log",'g',0),
    ("logs/_GCPA_PSP_orgdataset/test_GCPAPSPNet_2021-03-13 11:55:22.087377_5_CVC-ClinicDB.log",'r',0),
    ("logs/_GCPA_ASPP_orgdataset/test_GCPAASPPNet_2021-03-13 12:35:22.201283_5_CVC-ClinicDB.log",'g',0),

]

for path, color, ty in path_colors:
    data = (open(path,"r").read().split("\n"))
    if ty == 0:
        x = [int(data[i][-3:]) for i in range(2, len(data), 6) if len(data[i][-3:]) > 1]
        y = [(int(data[i][-3:]) - min_val) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3]
        # y = [(int(data[i][-21:-18]) - min_val) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3]
    elif ty == 1:
        x = [int(data[i][-3:]) for i in range(2, len(data), 5) if len(data[i][-3:]) > 1]
        y = [(int(data[i][-3:]) - min_val) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3]
        # y = [(int(data[i][-21:-18]) - min_val) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3]
    plt.bar(x, y, color=color)
