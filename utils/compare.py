
from matplotlib import pyplot as plt
min_val = 850

# Kvasir

# Res2Net, ResNet, Hardnet (CGNL)
path_colors = [
    ("logs/GCPA_GALDv4_orgdataset/test_2021-03-07 23:18:34.142658_GCPAGALDNetv4_5.log",'r',1,"Res2Net50"),
    ("logs/GCPA_GALDv5_orgdataset/test_2021-03-09 15:07:49.599382_GCPAGALDNetv5_5_Kvasir.log",'g',1,"ResNet50"),
    ("logs/GCPA_GALDv6_orgdataset/test_2021-03-10 01:15:09.192406_GCPAGALDNetv6_5_Kvasir.log",'b',1, "Hardnet"),
    # ("logs/GCPA_CCv3_orgdataset/test_GCPACCv3Net_2021-03-21 23:07:05.753043_5_Kvasir.log",'cyan',0),
]
path_colors1 = [
    ("logs/GCPA_GALDv4_orgdataset/test_2021-03-08 00:19:23.764247_GCPAGALDNetv4_5.log",'r',1,"Res2Net50"),
    ("logs/GCPA_GALDv5_orgdataset/test_2021-03-09 15:11:29.240565_GCPAGALDNetv5_5_CVC-ClinicDB.log",'g',1,"ResNet50"),
    ("logs/GCPA_GALDv6_orgdataset/test_2021-03-10 01:15:52.975074_GCPAGALDNetv6_5_CVC-ClinicDB.log",'b',1,"Hardnet"),
    # ("logs/GCPA_CCv3_orgdataset/test_GCPACCv3Net_2021-03-21 22:39:23.793020_5_CVC-ClinicDB.log",'cyan',0),
]

# => Choose hardnet

# full, remove SRM, remove HA,  remove mirror path
path_colors = [
    ("logs/GCPA_GALDv6_orgdataset/test_2021-03-10 01:15:09.192406_GCPAGALDNetv6_5_Kvasir.log",'r',1, "Full model"),
    ("logs/GCPA_GALDv7_orgdataset/test_2021-03-10 05:38:19.526538_GCPAGALDNetv7_5_Kvasir.log",'g',0, "Remove SR"),
    ("logs/GCPA_GALDv8_orgdataset/test_GCPAGALDNetv8_2021-03-10 10:58:50.415639_5_Kvasir.log",'b',0, "Remove SR & HA"),
    # ("logs/GCPA_GALDv9_orgdataset/test_GCPAGALDNetv9_2021-03-10 17:37:26.138396_5_Kvasir.log",'g',0),
]
path_colors1 = [
    ("logs/GCPA_GALDv6_orgdataset/test_2021-03-10 01:15:52.975074_GCPAGALDNetv6_5_CVC-ClinicDB.log",'r',1,"Full model"),
    ("logs/GCPA_GALDv7_orgdataset/test_2021-03-10 05:40:58.714650_GCPAGALDNetv7_5_CVC-ClinicDB.log",'g',0,"Remove SR"),
    ("logs/GCPA_GALDv8_orgdataset/test_GCPAGALDNetv8_2021-03-10 11:02:34.064225_5_CVC-ClinicDB.log",'b',0, "Remove SR & HA"),
    # ("logs/GCPA_GALDv9_orgdataset/test_GCPAGALDNetv9_2021-03-10 17:41:31.315643_5_CVC-ClinicDB.log",'g',0),
]

# ==> Just keep FAM (v8)

# ASPP, PSP, CGNL, CC, CC2, CC3AG (no HA)
path_colors = [
    ("logs/GCPA_ASPP_orgdataset/test_GCPAASPPNet_2021-03-12 06:53:41.989173_5_Kvasir.log",'r',2,"ASPP"),
    ("logs/GCPA_PSP_orgdataset/test_GCPAPSPNet_2021-03-12 23:10:22.569401_5_Kvasir.log",'lime',2,"PSP"),
    ("logs/GCPA_GALDv8_orgdataset/test_GCPAGALDNetv8_2021-03-10 10:58:50.415639_5_Kvasir.log",'b',2,"CGNL"),

    # ("logs/GCPA_CC_orgdataset/test_GCPACCNet_2021-03-19 00:22:43.880789_5_Kvasir.log",'cyan',0,"CC"),
    # ("logs/GCPACC2_orgdataset/test_GCPACC2Net_2021-04-26 13:09:02.305720_5_Kvasir.log",'y',0,"CC2"),
    # ("logs/GCPA_CC3AG_orgdataset/test_GCPACC3GANet_2021-04-26 23:21:25.071333_5_Kvasir.log",'m',0,"CC3AG"),
    # ("logs/SCWSCC2_orgdataset/test_SCWSCC2Net_2021-04-26 23:34:08.712965_5_Kvasir.log",'black',0,"FAM2-CC2"),
    ("logs/SCWSRCCA_orgdataset/test_SCWSRCCANet_2021-05-08 23:30:36.764196_5_Kvasir.log",'blue',0,"SWCWRCCA"),
    # ("logs/SCWS_LAMBDA_orgdataset/test_SCWSLambdaNet_2021-05-19 09:56:13.804233_5_Kvasir.log",'pink',0,"SCWSLambda"),

    # ("logs/SCWS_CC_orgdataset/test_SCWSCCNet_2021-04-10 14:19:20.497545_5_Kvasir.log",'gray',0),
    # ("logs/SCWS_PSP_orgdataset/test_SCWSPSPNet_2021-04-11 12:45:29.915980_5_Kvasir.log",'black',0),
    # ("logs/SCWS_PSP_Res_orgdataset/test_SCWSPSPResNet_2021-04-11 22:45:54.016374_5_Kvasir.log",'yellow',0),

]   
path_colors1 = [
    ("logs/GCPA_ASPP_orgdataset/test_GCPAASPPNet_2021-03-12 06:57:14.967380_5_CVC-ClinicDB.log",'r',2,"ASPP"),
    ("logs/GCPA_PSP_orgdataset/test_GCPAPSPNet_2021-03-12 23:10:44.953094_5_CVC-ClinicDB.log",'lime',2,"PSP"),
    ("logs/GCPA_GALDv8_orgdataset/test_GCPAGALDNetv8_2021-03-10 11:02:34.064225_5_CVC-ClinicDB.log",'b',2,"CGNL"),

    # ("logs/GCPA_CC_orgdataset/test_GCPACCNet_2021-03-19 00:25:33.623405_5_CVC-ClinicDB.log",'cyan',0,"CC"),
    # ("logs/GCPACC2_orgdataset/test_GCPACC2Net_2021-04-26 13:08:28.246676_5_CVC-ClinicDB.log",'y',0,"CC2"),
    # ("logs/GCPA_CC3AG_orgdataset/test_GCPACC3GANet_2021-04-26 23:20:36.218472_5_CVC-ClinicDB.log",'m',0,"CC3AG"),
    # ("logs/SCWSCC2_orgdataset/test_SCWSCC2Net_2021-04-26 23:22:20.325936_5_CVC-ClinicDB.log",'black',0,"FAM2-CC2"),
    ("logs/SCWSRCCA_orgdataset/test_SCWSRCCANet_2021-05-08 23:35:17.888119_5_CVC-ClinicDB.log",'blue',0,"SWCWRCCA"),
    # ("logs/SCWS_LAMBDA_orgdataset/test_SCWSLambdaNet_2021-05-17 21:24:41.396069_5_CVC-ClinicDB.log",'pink',0,"SCWSLambda"),
    
    # ("logs/SCWS_CC_orgdataset/test_SCWSCCNet_2021-04-10 14:14:52.827455_5_CVC-ClinicDB.log",'gray',0),
    # ("logs/SCWS_PSP_orgdataset/test_SCWSPSPNet_2021-04-11 12:45:57.410281_5_CVC-ClinicDB.log",'black',0),
    # ("logs/SCWS_PSP_Res_orgdataset/test_SCWSPSPResNet_2021-04-11 22:48:16.296961_5_CVC-ClinicDB.log",'yellow',0),
]
# Choose CC because the small size of model , PSP better a little



# GCE
path_colors = [
    # ("logs/GCPA_CC_orgdataset/test_GCPACCNet_2021-03-19 00:22:43.880789_5_Kvasir.log",'cyan',0,"CC-"),
    ("logs/GCPA_ASPP_orgdataset/test_GCPAASPPNet_2021-03-12 06:53:41.989173_5_Kvasir.log",'r',2,"ASPP"),
    ("logs/GCPA_PSP_orgdataset/test_GCPAPSPNet_2021-03-12 23:10:22.569401_5_Kvasir.log",'lime',2,"PSP"),
    ("logs/GCPA_GALDv8_orgdataset/test_GCPAGALDNetv8_2021-03-10 10:58:50.415639_5_Kvasir.log",'b',2,"CGNL"),
    ("logs/GCPARCCA_orgdataset/test_GCPARCCANet_2021-05-20 10:59:47.113276_5_Kvasir.log",'y',0,"CC"),

]   
path_colors1 = [
    # ("logs/GCPA_CC_orgdataset/test_GCPACCNet_2021-03-19 00:25:33.623405_5_CVC-ClinicDB.log",'cyan',0,"CC-"),
    ("logs/GCPA_ASPP_orgdataset/test_GCPAASPPNet_2021-03-12 06:57:14.967380_5_CVC-ClinicDB.log",'r',2,"ASPP"),
    ("logs/GCPA_PSP_orgdataset/test_GCPAPSPNet_2021-03-12 23:10:44.953094_5_CVC-ClinicDB.log",'lime',2,"PSP"),
    ("logs/GCPA_GALDv8_orgdataset/test_GCPAGALDNetv8_2021-03-10 11:02:34.064225_5_CVC-ClinicDB.log",'b',2,"CGNL"),
    ("logs/GCPARCCA_orgdataset/test_GCPARCCANet_2021-05-20 10:58:21.726931_5_CVC-ClinicDB.log",'y',0,"CC"),
]




# AGG
path_colors = [
    ("logs/GCPA_PSP_orgdataset/test_GCPAPSPNet_2021-03-12 23:10:22.569401_5_Kvasir.log",'lime',2,"FAM-PSP"),
    ("logs/GCPARCCA_orgdataset/test_GCPARCCANet_2021-05-20 10:59:47.113276_5_Kvasir.log",'y',0,"FAM-CC"),
    ("logs/SCWSRCCA_orgdataset/test_SCWSRCCANet_2021-05-08 23:30:36.764196_5_Kvasir.log",'y',0,"FAMv2-CC"),
    ("logs/SCWS_PSP_orgdataset/test_SCWSPSPNet_2021-04-11 12:45:29.915980_5_Kvasir.log",'lime',0,"FAMv2-PSP"),


]   
path_colors1 = [
    ("logs/GCPA_PSP_orgdataset/test_GCPAPSPNet_2021-03-12 23:10:44.953094_5_CVC-ClinicDB.log",'lime',2,"FAM-PSP"),
    ("logs/GCPARCCA_orgdataset/test_GCPARCCANet_2021-05-20 10:58:21.726931_5_CVC-ClinicDB.log",'y',0,"FAM-CC"),
    ("logs/SCWSRCCA_orgdataset/test_SCWSRCCANet_2021-05-08 23:35:17.888119_5_CVC-ClinicDB.log",'y',0,"FAMv2-CC"),
    ("logs/SCWS_PSP_orgdataset/test_SCWSPSPNet_2021-04-11 12:45:57.410281_5_CVC-ClinicDB.log",'lime',0,"FAMv2-PSP"),
]

# FAM, Add pranet1, pranet2 in GCPA, rm mirror(cgnl), ag, agv2 (PSP), SCWS
path_colors = [
    ("logs/GCPA_GALDv8_orgdataset/test_GCPAGALDNetv8_2021-03-10 10:58:50.415639_5_Kvasir.log",'b',2),
    # ("logs/PraNetvFAMGALD_orgdataset/test_PraNetvFAMGALD_2021-03-11 07:29:18.560009_5_Kvasir.log",'black',2),
    # ("logs/PraNetvFAM2GALD_orgdataset/test_PraNetvFAM2GALD_2021-03-11 07:50:40.020118_5_Kvasir.log",'cyan',2),
    # ("logs/GCPA_GALDv9_orgdataset/test_GCPAGALDNetv9_2021-03-10 17:37:26.138396_5_Kvasir.log",'g',0),
    ("logs/GCPA_PSPAG_orgdataset/test_GCPAPSPAGNet_2021-03-19254402_5_Kvasir.log",'r',0),
    # ("logs/GCPA_PSPAGv2_orgdataset/test_GCPAPSPAGv2Net_2021-03-20 07:23:12.386047_5_Kvasir.log",'y',0),
    ("logs/SCWS_CC_orgdataset/test_SCWSCCNet_2021-04-10 14:19:20.497545_5_Kvasir.log",'gray',0),
]
path_colors = [
    ("logs/GCPA_GALDv8_orgdataset/test_GCPAGALDNetv8_2021-03-10 11:02:34.064225_5_CVC-ClinicDB.log",'b',2),
    # ("logs/PraNetvFAMGALD_orgdataset/test_PraNetvFAMGALD_2021-03-11 07:29:58.243561_5_CVC-ClinicDB.log",'black',2),
    # ("logs/PraNetvFAM2GALD_orgdataset/test_PraNetvFAM2GALD_2021-03-11 08:10:47.353152_5_CVC-ClinicDB.log",'cyan',2),
    # ("logs/GCPA_GALDv9_orgdataset/test_GCPAGALDNetv9_2021-03-10 17:41:31.315643_5_CVC-ClinicDB.log",'g',0),
    ("logs/GCPA_PSPAG_orgdataset/test_GCPAPSPAGNet_2021-03-1904.476982_5_CVC-ClinicDB.log",'r',0),
    # ("logs/GCPA_PSPAGv2_orgdataset/test_GCPAPSPAGv2Net_2021-03-20 07:23:26.513804_5_CVC-ClinicDB.log",'y',0),
    ("logs/SCWS_CC_orgdataset/test_SCWSCCNet_2021-04-10 14:14:52.827455_5_CVC-ClinicDB.log",'gray',0),
]

# Kvasir, Clinic, ColonDB, ETIS, EndoScene in CC
path_colors = [
    ("logs/GCPA_CC_orgdataset/test_GCPACCNet_2021-03-19 00:22:43.880789_5_Kvasir.log",'r',0),
    ("logs/GCPA_CC_orgdataset/test_GCPACCNet_2021-03-19 00:25:33.623405_5_CVC-ClinicDB.log",'g',0),
    ("logs/GCPA_CC_orgdataset/test_GCPACCNet_2021-03-21 23:43:06.956350_5_CVC-ColonDB.log",'b',0),
    ("logs/GCPA_CC_orgdataset/test_GCPACCNet_2021-03-22 00:51:39.040925_5_ETIS-LaribPolypDB.log",'y',0),
    ("logs/GCPA_CC_orgdataset/test_GCPACCNet_2021-03-22 00:37:32.535939_5_CVC-300.log",'black',0),
]

# CC, CC with head, CC with Res2Net, CC with PraNet
path_colors = [
    ("logs/GCPA_CC_orgdataset/test_GCPACCNet_2021-03-19 00:22:43.880789_5_Kvasir.log",'r',0),
    ("logs/GCPA_CCv2_orgdataset/test_GCPACCv2Net_2021-03-20 01:03:44.439766_5_Kvasir.log",'g',0),
    ("logs/GCPA_CCv3_orgdataset/test_GCPACCv3Net_2021-03-21 23:07:05.753043_5_Kvasir.log",'b',0),
    ("logs/GCPA_CCv4_orgdataset/test_GCPACCv4Net_2021-03-22 10:21:07.720988_5_Kvasir.log",'y',0),
]
path_colors = [
    ("logs/GCPA_CC_orgdataset/test_GCPACCNet_2021-03-19 00:25:33.623405_5_CVC-ClinicDB.log",'r',0),
    ("logs/GCPA_CCv2_orgdataset/test_GCPACCv2Net_2021-03-20 01:04:19.572499_5_CVC-ClinicDB.log",'g',0),
    ("logs/GCPA_CCv3_orgdataset/test_GCPACCv3Net_2021-03-21 22:39:23.793020_5_CVC-ClinicDB.log",'b',0),
    ("logs/GCPA_CCv4_orgdataset/test_GCPACCv4Net_2021-03-22 10:20:49.310242_5_CVC-ClinicDB.log",'y',0),
]


# Kvasir 
path_colors = [
    ("logs/GCPA_ASPP_orgdataset/test_GCPAASPPNet_2021-03-12 06:53:41.989173_5_Kvasir.log",'r',0),
    ("logs/GCPA_PSP_orgdataset/test_GCPAPSPNet_2021-03-12 23:10:22.569401_5_Kvasir.log",'lime',0),

    ("logs/GCPA_GALDv9_orgdataset/test_GCPAGALDNetv9_2021-03-10 17:37:26.138396_5_Kvasir.log",'g',0),
    ("logs/GCPA_GALDv8_orgdataset/test_GCPAGALDNetv8_2021-03-10 10:58:50.415639_5_Kvasir.log",'b',0),
    ("logs/GCPA_GALDv7_orgdataset/test_2021-03-10 05:38:19.526538_GCPAGALDNetv7_5_Kvasir.log",'m',0),
    ("logs/GCPA_orgdataset/test_2021-03-08 22:14:28.523723_GCPANet_5.log",'orange',1),

    ("logs/PraNetvFAM2GALD_orgdataset/test_PraNetvFAM2GALD_2021-03-11 07:50:40.020118_5_Kvasir.log",'cyan',0),
    ("logs/GCPA_GALDv6_orgdataset/test_2021-03-10 01:15:09.192406_GCPAGALDNetv6_5_Kvasir.log",'y',1),
    ("logs/GCPA_GALDv4_orgdataset/test_2021-03-07 23:18:34.142658_GCPAGALDNetv4_5.log",'black',1),
    ("logs/pranetGALD_orgdataset/test_2021-03-06 15:21:49.319834_PraNetvGALD_3.log",'brown',1),
    ("logs/GCPA_GALDv5_orgdataset/test_2021-03-09 15:07:49.599382_GCPAGALDNetv5_5_Kvasir.log",'gray',1),
    ("logs/PraNetvFAMGALD_orgdataset/test_PraNetvFAMGALD_2021-03-11 07:29:18.560009_5_Kvasir.log",'white',0),
]
# Clinic
path_colors = [
    # ("logs/GCPA_GALDv5_orgdataset/test_2021-03-09 15:11:29.240565_GCPAGALDNetv5_5_CVC-ClinicDB.log",'gray',1),
    ("logs/GCPA_GALDv8_orgdataset/test_GCPAGALDNetv8_2021-03-10 11:02:34.064225_5_CVC-ClinicDB.log",'b',0),
    ("logs/GCPA_PSP_orgdataset/test_GCPAPSPNet_2021-03-12 23:10:44.953094_5_CVC-ClinicDB.log",'lime',0),
    # ("logs/GCPA_GALDv6_orgdataset/test_2021-03-10 01:15:52.975074_GCPAGALDNetv6_5_CVC-ClinicDB.log",'y',1),
    ("logs/GCPA_GALDv7_orgdataset/test_2021-03-10 05:40:58.714650_GCPAGALDNetv7_5_CVC-ClinicDB.log",'m',0),
    # ("logs/GCPA_ASPP_orgdataset/test_GCPAASPPNet_2021-03-12 06:57:14.967380_5_CVC-ClinicDB.log",'r',0),
    # ("logs/GCPA_GALDv9_orgdataset/test_GCPAGALDNetv9_2021-03-10 17:41:31.315643_5_CVC-ClinicDB.log",'g',0),
    # ("logs/GCPA_GALDv4_orgdataset/test_2021-03-08 00:19:23.764247_GCPAGALDNetv4_5.log",'black',1),
    # ("logs/pranetGALD_orgdataset/test_2021-03-06 15:29:40.884150_PraNetvGALD_3.log",'brown',1),
    # ("logs/PraNetvFAM2GALD_orgdataset/test_PraNetvFAM2GALD_2021-03-11 08:10:47.353152_5_CVC-ClinicDB.log",'cyan',0),
    # ("logs/PraNetvFAMGALD_orgdataset/test_PraNetvFAMGALD_2021-03-11 07:29:58.243561_5_CVC-ClinicDB.log",'white',0),
]

# PSPAG
path_colors = [
    ("logs/GCPA_PSPAG_orgdataset/test_GCPAPSPAGNet_2021-03-19254402_5_Kvasir.log",'r',0),
    ("logs/GCPA_PSPAG_orgdataset/test_GCPAPSPAGNet_2021-03-1904.476982_5_CVC-ClinicDB.log",'r',0),
    ("logs/GCPA_PSP_orgdataset/test_GCPAPSPNet_2021-03-12 23:10:22.569401_5_Kvasir.log",'g',0),
    ("logs/GCPA_PSP_orgdataset/test_GCPAPSPNet_2021-03-12 23:10:44.953094_5_CVC-ClinicDB.log",'g',0),
    ("logs/GCPA_CC_orgdataset/test_GCPACCNet_2021-03-19 00:22:43.880789_5_Kvasir.log",'b',0),
    ("logs/GCPA_CC_orgdataset/test_GCPACCNet_2021-03-19 00:25:33.623405_5_CVC-ClinicDB.log",'b',0),
    ("logs/GCPA_CCv2_orgdataset/test_GCPACCv2Net_2021-03-20 01:03:44.439766_5_Kvasir.log",'y',0),
    ("logs/GCPA_CCv2_orgdataset/test_GCPACCv2Net_2021-03-20 01:04:19.572499_5_CVC-ClinicDB.log",'y',0),
]

# 0 - new
# 1 - old 
# 2 - new but fake

for path, color, ty in path_colors:
    data = (open(path,"r").read().split("\n"))
    if ty == 0:
        x = [int(data[i][-3:]) for i in range(2, len(data), 6) if len(data[i][-3:]) > 1]
        y = [(int(data[i][-3:])) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3]
        # y = [(int(data[i][-21:-18]) - min_val) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3]
    elif ty == 1:
        x = [int(data[i][-3:]) for i in range(2, len(data), 5) if len(data[i][-3:]) > 1]
        y = [(int(data[i][-3:])) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3]
        # y = [(int(data[i][-21:-18]) - min_val) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3]
    elif ty == 2:
        x = [int(data[i][-3:])//3+134 for i in range(2, len(data), 6) if len(data[i][-3:]) > 1]
        y = [(int(data[i][-3:])) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3]

    plt.plot(x, y, color=color)


from scipy.ndimage.filters import gaussian_filter1d

fig, axs = plt.subplots(2, 1,constrained_layout=True)
# fig.tight_layout()
# 0 - new
# 1 - old 
# 2 - new but fake

for path, color, ty, label in path_colors:
    data = (open(path,"r").read().split("\n"))
    if ty == 0:
        x = [int(data[i][-3:]) for i in range(2, len(data), 6) if len(data[i][-3:]) > 1]
        y = [(int(data[i][-3:])) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3]
        # y = [(int(data[i][-21:-18]) - min_val) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3]
    elif ty == 1:
        x = [int(data[i][-3:]) for i in range(2, len(data), 5) if len(data[i][-3:]) > 1]
        y = [(int(data[i][-3:])) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3]
        # y = [(int(data[i][-21:-18]) - min_val) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3]
    elif ty == 2:
        x = [int(data[i][-3:])//3+134 for i in range(2, len(data), 6) if len(data[i][-3:]) > 1]
        y = [(int(data[i][-3:])) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3]

    elif ty == 3:
        x = [int(data[i][-3:])//3+134 for i in range(2, len(data), 5) if len(data[i][-3:]) > 1]
        y = [(int(data[i][-3:])) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3]
    if("v2" in label):
        axs[0].plot(x[-40:], gaussian_filter1d(y[-40:], sigma=sigma),label = label, color=color,linestyle='dashed')
    else:
        if (sigma == 0):
            axs[0].plot(x[-40:], y[-40:],label = label, color=color)
        else:
            axs[0].plot(x[-40:], gaussian_filter1d(y[-40:], sigma=sigma),label = label, color=color)
    axs[0].set_title("Kvasir")
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("DSC(%)") 
    axs[0].legend(bbox_to_anchor =(1, 1.6), ncol = 4)
    # axs[0].legend(loc='upper right', frameon=False)

for path, color, ty, label in path_colors1:
    data = (open(path,"r").read().split("\n"))
    if ty == 0:
        x = [int(data[i][-3:]) for i in range(2, len(data), 6) if len(data[i][-3:]) > 1]
        y = [(int(data[i][-3:])) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3]
        # y = [(int(data[i][-21:-18]) - min_val) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3]
    elif ty == 1:
        x = [int(data[i][-3:]) for i in range(2, len(data), 5) if len(data[i][-3:]) > 1]
        y = [(int(data[i][-3:])) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3]
        # y = [(int(data[i][-21:-18]) - min_val) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3]
    elif ty == 2:
        x = [int(data[i][-3:])//3+134 for i in range(2, len(data), 6) if len(data[i][-3:]) > 1]
        y = [(int(data[i][-3:])) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3]

    elif ty == 3:
        x = [int(data[i][-3:])//3+134 for i in range(2, len(data), 5) if len(data[i][-3:]) > 1]
        y = [(int(data[i][-3:])) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3]
    if("v2" in label):
        axs[1].plot(x[-40:], gaussian_filter1d(y[-40:], sigma=sigma),label = label, color=color,linestyle='dashed')
    else:
        if (sigma == 0):
            axs[1].plot(x[-40:], y[-40:],label = label, color=color)
        else:
            axs[1].plot(x[-40:], gaussian_filter1d(y[-40:], sigma=sigma),label = label, color=color)
    
    axs[1].set_title("Clinic")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("DSC(%)")
    # axs[1].legend(bbox_to_anchor =(0.75, 1.15), ncol = 2)
    # axs[1].legend(loc='best')








fig, axs = plt.subplots(2, 1,constrained_layout=True)
# fig.tight_layout()
# 0 - new
# 1 - old 
# 2 - new but fake

for path, color, ty, label in path_colors:
    data = (open(path,"r").read().split("\n"))
    if ty == 0:
        x = [int(data[i][-3:]) for i in range(2, len(data), 6) if len(data[i][-3:]) > 1]
        # y = [(int(data[i][-3:])) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3]
        y = [(int(data[i][-21:-18])) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3]
    elif ty == 1:
        x = [int(data[i][-3:]) for i in range(2, len(data), 5) if len(data[i][-3:]) > 1]
        y = [(int(data[i][-3:])) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3]
        # y = [(int(data[i][-21:-18]) - min_val) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3]
    elif ty == 2:
        x = [int(data[i][-3:])//3+134 for i in range(2, len(data), 6) if len(data[i][-3:]) > 1]
        # y = [(int(data[i][-3:])) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3]
        y = [(int(data[i][-21:-18])) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3]

    elif ty == 3:
        x = [int(data[i][-3:])//3+134 for i in range(2, len(data), 5) if len(data[i][-3:]) > 1]
        y = [(int(data[i][-3:])) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3]
    axs[0].plot(x, y,label = label, color=color)
    axs[0].set_title("Kvasir")
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("DSC(%)") 
    axs[0].legend(bbox_to_anchor =(1, 1.6), ncol = 4)
    # axs[0].legend(loc='upper right', frameon=False)

for path, color, ty, label in path_colors1:
    data = (open(path,"r").read().split("\n"))
    if ty == 0:
        x = [int(data[i][-3:]) for i in range(2, len(data), 6) if len(data[i][-3:]) > 1]
        # y = [(int(data[i][-3:])) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3]
        y = [(int(data[i][-21:-18])) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3]
    elif ty == 1:
        x = [int(data[i][-3:]) for i in range(2, len(data), 5) if len(data[i][-3:]) > 1]
        y = [(int(data[i][-3:])) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3]
        # y = [(int(data[i][-21:-18])) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3]
    elif ty == 2:
        x = [int(data[i][-3:])//3+134 for i in range(2, len(data), 6) if len(data[i][-3:]) > 1]
        # y = [(int(data[i][-3:])) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3]
        y = [(int(data[i][-21:-18])) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3]

    elif ty == 3:
        x = [int(data[i][-3:])//3+134 for i in range(2, len(data), 5) if len(data[i][-3:]) > 1]
        y = [(int(data[i][-3:])) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3]
    axs[1].plot(x, y,label = label, color=color)
    
    axs[1].set_title("Clinic")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("DSC(%)")
    # axs[1].legend(bbox_to_anchor =(0.75, 1.15), ncol = 2)
    # axs[1].legend(loc='best')
    


# Train 150 epoch
path_colors = [
    # ("logs/_GCPA_ASPP_orgdataset/test_GCPAASPPNet_2021-03-13 12:34:26.008870_5_Kvasir.log",'r',0),
    # ("logs/_GCPA_PSP_orgdataset/test_GCPAPSPNet_2021-03-13 11:54:56.156760_5_Kvasir.log",'g',0),
    ("logs/_GCPA_PSP_orgdataset/test_GCPAPSPNet_2021-03-13 11:55:22.087377_5_CVC-ClinicDB.log",'r',0),
    ("logs/_GCPA_ASPP_orgdataset/test_GCPAASPPNet_2021-03-13 12:35:22.201283_5_CVC-ClinicDB.log",'g',0),

]
                            
# PSPSmall all 30 epoch
path_colors = [
    ("logs/noaug_GCPAPSPSmallNet_oridataset/test_GCPAPSPSmallNet_2021-03-16 12:18:17.107010_5_CVC-ColonDB.log",'r',0),
    ("logs/noaug_GCPAPSPSmallNet_oridataset/test_GCPAPSPSmallNet_2021-03-16 12:17:36.268311_5_CVC-300.log",'g',0),
    ("logs/noaug_GCPAPSPSmallNet_oridataset/test_GCPAPSPSmallNet_2021-03-16 12:16:52.329484_5_ETIS-LaribPolypDB.log",'b',0),
    ("logs/noaug_GCPAPSPSmallNet_oridataset/test_GCPAPSPSmallNet_2021-03-16 12:12:19.614638_5_Kvasir.log",'black',0),
    ("logs/noaug_GCPAPSPSmallNet_oridataset/test_GCPAPSPSmallNet_2021-03-16 12:14:08.121800_5_CVC-ClinicDB.log",'y',0),
]

for path, color, ty in path_colors:
    data = (open(path,"r").read().split("\n"))
    if ty == 0:
        x = [int(data[i].split(" ")[-1]) for i in range(2, len(data), 6) if len(data[i][-3:]) > 1]
        y = [(int(data[i][-3:]) - min_val) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3]
        # y = [(int(data[i][-21:-18]) - min_val) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3]
    elif ty == 1:
        x = [int(data[i].split(" ")[-1]) for i in range(2, len(data), 5) if len(data[i][-3:]) > 1]
        y = [(int(data[i][-3:]) - min_val) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3]
        # y = [(int(data[i][-21:-18]) - min_val) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3]
    plt.plot(x, y, color=color)



path = "/Users/brown/code/polyp_segmentation/logs/train_GCPAPSPNet_2021-03-16 21:38:21.872529_5.log"
data = (open(path,"r").read().split("\n"))
# EPOCH
x = [int(data[i].split("Epoch")[-1][2:5]) for i in range(8, 511, 5)]
x.extend([int(data[i].split("Epoch")[-1][2:5]) for i in range(513, len(data), 6)])

# loss_val
y = [float(data[i].split("Epoch")[1][-7:-1]) for i in range(8, 511, 5)]
y.extend([float(data[i].split("Epoch")[1][-7:-1]) for i in range(513, len(data), 6)])

# f1
z = [float(data[i][-5:]) for i in range(9, 511, 5)]
z.extend([float(data[i][-5:]) for i in range(514, len(data), 6)])

# LOSS ALL
t = []
for i in range(7, 511, 5):
    # t.append(float(data[i-1][-7:-1]))
    t.append(float(data[i][-7:-1]))
for i in range(512, len(data)-1, 6):
    # t.append(float(data[i-1][-7:-1]))
    t.append(float(data[i][-7:-1]))

# Loss 2
w = []
for i in range(7, 511, 5):
    # t.append(float(data[i-1][-7:-1]))
    w.append(float(data[i].split("loss_record2")[-1][2:8]))
for i in range(512, len(data)-1, 6):
    # t.append(float(data[i-1][-7:-1]))
    w.append(float(data[i].split("loss_record2")[-1][2:8]))
    
plt.plot(x, y, color="r")
plt.plot(x, z, color="g")
plt.plot(x, w, color="y")




path = "logs\\train_GCPACCNet_2021-03-18.312114_5"
data = (open(path,"r").read().split("\n"))
# EPOCH
x = [int(data[i].split("Epoch")[-1][2:5]) for i in range(10, 1066, 7)]
x.extend([int(data[i].split("Epoch")[-1][2:5]) for i in range(1067, len(data), 8)])

# loss_val
y = [float(data[i].split("Epoch")[1][-7:-1]) for i in range(10, 1066, 7)]
y.extend([float(data[i].split("Epoch")[1][-7:-1]) for i in range(1067, len(data), 8)])

# f1
z = [float(data[i][-5:]) for i in range(12, 1068, 7)]
z.extend([float(data[i][-5:]) for i in range(1069, len(data), 8)])



path = "logs/noaug_GCPACCNet_orgdataset/train_GCPACCNet_2021-03-20 07:56:23.738934_5.log"
data = (open(path,"r").read().split("\n"))

x = [int(data[i].split("Epoch")[-1][2:5]) for i in range(10, 232, 7)]
x.extend([int(data[i].split("Epoch")[-1][2:5]) for i in range(233, len(data)-2, 8)])
y = [float(data[i][-7:-1]) for i in range(10, 232, 7)]
y.extend([float(data[i][-7:-1]) for i in range(233, len(data)-1, 8)])
# f1
z = [float(data[i][-5:]) for i in range(11, 234, 7)]
z.extend([float(data[i][-5:]) for i in range(236, len(data)-1, 8)])
# loss2
w = [float(data[i][-6:-1]) for i in range(11, 234, 7)]
w.extend([float(data[i][-6:-1]) for i in range(235, len(data)-1, 8)])
plt.plot(x, y, color="r")
plt.plot(x, z, color="g")

plt.plot(y[25:], z[25:], 'r--', color="g")


# CCv3 aug
path = "logs/train_GCPACCv3Net_2021-03-21 11:24:40.935707_5.log"
data = (open(path,"r").read().split("\n"))

x = [int(data[i].split("Epoch")[-1][2:5]) for i in range(9, len(data)-5, 6)]
y = [float(data[i][-7:-1]) for i in range(9, len(data)-4, 6)]
# f1
z = [float(data[i][-5:]) for i in range(10, len(data)-4, 6)]
# loss2
plt.plot(x, y, color="r")
plt.plot(x, z, color="g")

plt.plot(y[25:], z[25:], 'r--', color="g")



import glob

path = "data/kvasir-instrument/train.txt"
train = open(path,"r").read().split("\n")
path = "data/kvasir-instrument/test.txt"
test = open(path,"r").read().split("\n")
images = glob.glob("data/kvasir-instrument/images/*")
import os
for image in images:

    if(os.path.splitext(os.path.basename(image))[0] in train):
        cmd = f'cp {image} data/kvasir-instrument/traindataset/images'
    else:
        cmd = f'cp {image} data/kvasir-instrument/testdataset/images'
    print(cmd)
    os.system(cmd)

masks = glob.glob("data/kvasir-instrument/masks/*")
for mask in masks:

    if(os.path.splitext(os.path.basename(mask))[0] in train):
        cmd = f'cp {mask} data/kvasir-instrument/traindataset/masks'
    else:
        cmd = f'cp {mask} data/kvasir-instrument/testdataset/masks'
    os.system(cmd)