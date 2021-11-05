
from matplotlib import pyplot as plt
min_val = 850

# Kvasir

# Res2Net, ResNet, Hardnet (CGNL)
path_colors = [
    ("logs/GCPA_GALDv4_orgdataset/test_2021-03-07 23:18:34.142658_GCPAGALDNetv4_5.log",'r',3,"Res2Net50"),
    ("logs/GCPA_GALDv5_orgdataset/test_2021-03-09 15:07:49.599382_GCPAGALDNetv5_5_Kvasir.log",'g',4,"ResNet50"),
    ("logs/GCPA_GALDv6_orgdataset/test_2021-03-10 01:15:09.192406_GCPAGALDNetv6_5_Kvasir.log",'b',3, "Hardnet"),
    # ("logs/GCPA_CCv3_orgdataset/test_GCPACCv3Net_2021-03-21 23:07:05.753043_5_Kvasir.log",'cyan',0),
]
path_colors1 = [
    ("logs/GCPA_GALDv4_orgdataset/test_2021-03-08 00:19:23.764247_GCPAGALDNetv4_5.log",'r',3,"Res2Net50"),
    ("logs/GCPA_GALDv5_orgdataset/test_2021-03-09 15:11:29.240565_GCPAGALDNetv5_5_CVC-ClinicDB.log",'g',4,"ResNet50"),
    ("logs/GCPA_GALDv6_orgdataset/test_2021-03-10 01:15:52.975074_GCPAGALDNetv6_5_CVC-ClinicDB.log",'b',3,"Hardnet"),
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
    ("logs/GCPA_CC_orgdataset/test_GCPACCNet_2021-03-19 00:22:43.880789_5_Kvasir.log",'y',0,"CC"),
    ("logs/GCPA_ASPP_orgdataset/test_GCPAASPPNet_2021-03-12 06:53:41.989173_5_Kvasir.log",'r',2,"ASPP"),
    ("logs/GCPA_PSP_orgdataset/test_GCPAPSPNet_2021-03-12 23:10:22.569401_5_Kvasir.log",'lime',2,"PSP"),
    ("logs/GCPA_GALDv8_orgdataset/test_GCPAGALDNetv8_2021-03-10 10:58:50.415639_5_Kvasir.log",'b',2,"CGNL"),
    # ("logs/GCPARCCA_orgdataset/test_GCPARCCANet_2021-05-20 10:59:47.113276_5_Kvasir.log",'y',0,"CC"),

]   
path_colors1 = [
    ("logs/GCPA_CC_orgdataset/test_GCPACCNet_2021-03-19 00:25:33.623405_5_CVC-ClinicDB.log",'y',0,"CC"),
    ("logs/GCPA_ASPP_orgdataset/test_GCPAASPPNet_2021-03-12 06:57:14.967380_5_CVC-ClinicDB.log",'r',2,"ASPP"),
    ("logs/GCPA_PSP_orgdataset/test_GCPAPSPNet_2021-03-12 23:10:44.953094_5_CVC-ClinicDB.log",'lime',2,"PSP"),
    ("logs/GCPA_GALDv8_orgdataset/test_GCPAGALDNetv8_2021-03-10 11:02:34.064225_5_CVC-ClinicDB.log",'b',2,"CGNL"),
    # ("logs/GCPARCCA_orgdataset/test_GCPARCCANet_2021-05-20 10:58:21.726931_5_CVC-ClinicDB.log",'y',0,"CC"),
]




# FAM
path_colors = [
    ("logs/GCPA_PSPv2_orgdataset/test_GCPAPSPNet_2021-09-23 20:08:50.363558_5_Kvasir.log",'lime',0,"FAM-PSP"),
    # ("logs/GCPA_PSP_orgdataset/test_GCPAPSPNet_2021-03-12 23:10:22.569401_5_Kvasir.log",'lime',2,"FAM-PSP"),
    # ("logs/GCPARCCA_orgdataset/test_GCPARCCANet_2021-05-20 10:59:47.113276_5_Kvasir.log",'y',0,"FAM-CC"),
    ("logs/GCPA_CC_orgdataset/test_GCPACCNet_2021-03-19 00:22:43.880789_5_Kvasir.log",'y',0,"FAM-CC"),
    ("logs/SCWSRCCA_orgdataset/test_SCWSRCCANet_2021-05-08 23:30:36.764196_5_Kvasir.log",'y',0,"FAMv2-CC"),
    # ("logs/SCWSRCCAv2_orgdataset/test_SCWSRCCANet_2021-05-08 23:30:36.764196_5_Kvasir.log",'y',0,"FAMv2-CC"),
    ("logs/SCWS_PSP_orgdataset/test_SCWSPSPNet_2021-04-11 12:45:29.915980_5_Kvasir.log",'lime',0,"FAMv2-PSP"),

]   
path_colors1 = [
    # ("logs/GCPA_PSP_orgdataset/test_GCPAPSPNet_2021-03-12 23:10:44.953094_5_CVC-ClinicDB.log",'lime',2,"FAM-PSP"),
    ("logs/GCPA_PSPv2_orgdataset/test_GCPAPSPNet_2021-09-23 20:11:55.415501_5_CVC-ClinicDB.log",'lime',0,"FAM-PSP"),
    # ("logs/GCPARCCA_orgdataset/test_GCPARCCANet_2021-05-20 10:58:21.726931_5_CVC-ClinicDB.log",'y',0,"FAM-CC"),
    ("logs/GCPA_CC_orgdataset/test_GCPACCNet_2021-03-19 00:25:33.623405_5_CVC-ClinicDB.log",'y',0,"FAM-CC"),
    ("logs/SCWSRCCA_orgdataset/test_SCWSRCCANet_2021-05-08 23:35:17.888119_5_CVC-ClinicDB.log",'y',0,"FAMv2-CC"),
    # ("logs/SCWSRCCAv2_orgdataset/test_SCWSRCCANet_2021-05-08 23:35:17.888119_5_CVC-ClinicDB.log",'y',0,"FAMv2-CC"),
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
sigma = 1
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
    elif ty == 4:
        x = [int(data[i][-3:])//3+140 for i in range(2, len(data), 5) if len(data[i][-3:]) > 1]
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
    elif ty == 4:
        x = [int(data[i][-3:])//3+140 for i in range(2, len(data), 5) if len(data[i][-3:]) > 1]
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




# BACKBONE
path_colors = [
    [
        ("logs/GCPA_GALDv4_orgdataset/test_2021-03-07 23:18:34.142658_GCPAGALDNetv4_5.log",'r',3,"Res2Net50"),
        ("logs/GCPAGALDNetv4_v1/test_GCPAGALDNetv5_2021-10-14 21:30:55.846337_5_Kvasir.log",'r',0,"Res2Net50"),
        ("logs/GCPAGALDNetv4_v2/test_GCPAGALDNetv4_2021-10-14 21:55:48.262176_5_Kvasir.log",'r',0,"Res2Net50"),
        ("logs/GCPAGALDNetv4_v3/test_GCPAGALDNetv4_2021-10-14 22:01:42.448293_5_Kvasir.log",'r',0,"Res2Net50"),

    ],
    [
        ("logs/GCPA_GALDv5_orgdataset/test_2021-03-09 15:07:49.599382_GCPAGALDNetv5_5_Kvasir.log",'g',4,"ResNet50"),
        ("logs/GCPAGALDNetv5_v1/test_GCPAGALDNetv5_2021-10-14 21:18:37.815809_5_Kvasir.log",'g',0,"ResNet50"),
        ("logs/GCPAGALDNetv5_v2/test_GCPAGALDNetv5_2021-10-14 21:30:55.846337_5_Kvasir.log",'g',0,"ResNet50"),
        ("logs/GCPAGALDNetv5_v3/test_GCPAGALDNetv5_2021-10-14 21:34:49.201484_5_Kvasir.log",'g',0,"ResNet50"),
        ("logs/GCPAGALDNetv5_v4/test_GCPAGALDNetv5_2021-10-14 21:43:50.400637_5_Kvasir.log",'g',0,"ResNet50"),
        ("logs/GCPAGALDNetv5_v5/test_GCPAGALDNetv5_2021-10-19 20:25:10.074622_5_Kvasir.log",'g',0,"ResNet50"),
        ("logs/GCPAGALDNetv5_v6/test_GCPAGALDNetv5_2021-10-19 20:34:30.112608_5_Kvasir.log",'g',0,"ResNet50"),
    ],
#           [
#     ("logs/GCPA_GALDv8_orgdataset/test_GCPAGALDNetv8_2021-03-10 10:58:50.415639_5_Kvasir.log",'b',2,"CGNL"),
#     ("logs/GCPAGALDNetv8_kfoldv1/test_GCPAGALDNetv8_2021-09-29 01:18:57.137598_5_Kvasir.log",'b',0,"CGNL"),
#     ("logs/GCPAGALDNetv8_kfoldv2/test_GCPAGALDNetv8_2021-09-29 01:20:17.857876_5_Kvasir.log",'b',0,"CGNL"),
#     ("logs/GCPAGALDNetv8_kfoldv3/test_GCPAGALDNetv8_2021-09-29 01:42:52.984295_5_Kvasir.log",'b',0,"CGNL"),
#     ("logs/GCPAGALDNetv8_kfoldv4/test_GCPAGALDNetv8_2021-09-29 01:47:01.257980_5_Kvasir.log",'b',0,"CGNL"),
#    ],
             [
    ("logs/GCPAGALDNetv6_v1/test_GCPAGALDNetv6_2021-10-19 17:10:58.369682_5_Kvasir.log",'b',0,"Hardnet"),
    ("logs/GCPAGALDNetv6_v2/test_GCPAGALDNetv6_2021-10-19 17:57:56.604809_5_Kvasir.log",'b',0,"Hardnet"),
    ("logs/GCPAGALDNetv6_v3/test_GCPAGALDNetv6_2021-10-19 18:16:57.332534_5_Kvasir.log",'b',0,"Hardnet"),
    ("logs/GCPAGALDNetv6_v4/test_GCPAGALDNetv6_2021-10-19 19:59:14.671645_5_Kvasir.log",'b',0,"Hardnet"),
    ("logs/GCPA_GALDv6_orgdataset/test_2021-03-10 01:15:09.192406_GCPAGALDNetv6_5_Kvasir.log",'b',3,"Hardnet"),
   ]
    # ("logs/GCPA_GALDv6_orgdataset/test_2021-03-10 01:15:09.192406_GCPAGALDNetv6_5_Kvasir.log",'b',3, "Hardnet"),
]
path_colors1 = [
    [
        ("logs/GCPA_GALDv4_orgdataset/test_2021-03-08 00:19:23.764247_GCPAGALDNetv4_5.log",'r',3,"Res2Net50"),
        ("logs/GCPAGALDNetv4_v1/test_GCPAGALDNetv4_2021-10-14 21:50:45.889593_5_CVC-ClinicDB.log",'r',0,"Res2Net50"),
        ("logs/GCPAGALDNetv4_v2/test_GCPAGALDNetv4_2021-10-14 21:55:54.851841_5_CVC-ClinicDB.log",'r',0,"Res2Net50"),
        ("logs/GCPAGALDNetv4_v3/test_GCPAGALDNetv4_2021-10-14 22:05:39.518460_5_CVC-ClinicDB.log",'r',0,"Res2Net50"),
    ],
    [
        ("logs/GCPA_GALDv5_orgdataset/test_2021-03-09 15:11:29.240565_GCPAGALDNetv5_5_CVC-ClinicDB.log",'g',4,"ResNet50"),
        ("logs/GCPAGALDNetv5_v1/test_GCPAGALDNetv5_2021-10-14 21:27:19.784968_5_CVC-ClinicDB.log",'g',0,"ResNet50"),
        ("logs/GCPAGALDNetv5_v2/test_GCPAGALDNetv5_2021-10-14 21:32:00.220016_5_CVC-ClinicDB.log",'g',0,"ResNet50"),
        ("logs/GCPAGALDNetv5_v3/test_GCPAGALDNetv5_2021-10-14 21:37:12.006013_5_CVC-ClinicDB.log",'g',0,"ResNet50"),
        ("logs/GCPAGALDNetv5_v4/test_GCPAGALDNetv5_2021-10-14 21:44:03.658645_5_CVC-ClinicDB.log",'g',0,"ResNet50"),
        ("logs/GCPAGALDNetv5_v5/test_GCPAGALDNetv5_2021-10-19 20:25:22.616781_5_CVC-ClinicDB.log",'g',0,"ResNet50"),
        ("logs/GCPAGALDNetv5_v6/test_GCPAGALDNetv5_2021-10-19 20:34:41.526411_5_CVC-ClinicDB.log",'g',0,"ResNet50"),
    ],
#           [
#     ("logs/GCPA_GALDv8_orgdataset/test_GCPAGALDNetv8_2021-03-10 10:58:50.415639_5_Kvasir.log",'b',2,"CGNL"),
#     ("logs/GCPAGALDNetv8_kfoldv1/test_GCPAGALDNetv8_2021-09-29 01:18:57.137598_5_Kvasir.log",'b',0,"CGNL"),
#     ("logs/GCPAGALDNetv8_kfoldv2/test_GCPAGALDNetv8_2021-09-29 01:20:17.857876_5_Kvasir.log",'b',0,"CGNL"),
#     ("logs/GCPAGALDNetv8_kfoldv3/test_GCPAGALDNetv8_2021-09-29 01:42:52.984295_5_Kvasir.log",'b',0,"CGNL"),
#     ("logs/GCPAGALDNetv8_kfoldv4/test_GCPAGALDNetv8_2021-09-29 01:47:01.257980_5_Kvasir.log",'b',0,"CGNL"),
#    ],
      [
    ("logs/GCPAGALDNetv6_v1/test_GCPAGALDNetv6_2021-10-19 17:11:46.641428_5_CVC-ClinicDB.log",'b',0,"CGNL"),
    ("logs/GCPAGALDNetv6_v2/test_GCPAGALDNetv6_2021-10-19 17:58:06.679984_5_CVC-ClinicDB.log",'b',0,"CGNL"),
    ("logs/GCPAGALDNetv6_v3/test_GCPAGALDNetv6_2021-10-19 18:17:06.347776_5_CVC-ClinicDB.log",'b',0,"CGNL"),
    ("logs/GCPAGALDNetv6_v4/test_GCPAGALDNetv6_2021-10-19 19:59:20.464595_5_CVC-ClinicDB.log",'b',0,"CGNL"),
    ("logs/GCPA_GALDv6_orgdataset/test_2021-03-10 01:15:52.975074_GCPAGALDNetv6_5_CVC-ClinicDB.log",'b',3,"Hardnet"),
   ]

    # ("logs/GCPA_GALDv6_orgdataset/test_2021-03-10 01:15:52.975074_GCPAGALDNetv6_5_CVC-ClinicDB.log",'b',3,"Hardnet"),
]
# GCE
path_colors = [
    [
        ("logs/GCPAPSPResNet_v0/test_GCPAPSPResNet_2021-11-04 22:08:17.103530_5_Kvasir.log",'r',0,"PSP"),
        ("logs/GCPAPSPResNet_v1/test_GCPAPSPResNet_2021-11-05 00:49:44.530899_5_Kvasir.log",'r',0,"PSP"),
        ("logs/GCPAPSPResNet_v2/test_GCPAPSPResNet_2021-11-05 01:24:13.751833_5_Kvasir.log",'r',0,"PSP"),
        ("logs/GCPAPSPResNet_v3/test_GCPAPSPResNet_2021-11-05 02:07:24.260250_5_Kvasir.log",'r',0,"PSP"),
        ("logs/GCPAPSPResNet_v4/test_GCPAPSPResNet_2021-11-05 02:27:38.179117_5_Kvasir.log",'r',0,"PSP"),

    ],
    [
        ("logs/GCPAASPPResNet_v0/test_GCPAASPPResNet_2021-11-05 09:44:32.056548_5_Kvasir.log",'g',0,"ASPP"),
        ("logs/GCPAASPPResNet_v1/test_GCPAASPPResNet_2021-11-05 10:11:53.600390_5_Kvasir.log",'g',0,"ASPP"),
        ("logs/GCPAASPPResNet_v2/test_GCPAASPPResNet_2021-11-05 10:42:54.317850_5_Kvasir.log",'g',0,"ASPP"),
        ("logs/GCPAASPPResNet_v3/test_GCPAASPPResNet_2021-11-05 11:33:25.769853_5_Kvasir.log",'g',0,"ASPP"),
        ("logs/GCPAASPPResNet_v4/test_GCPAASPPResNet_2021-11-05 15:42:25.791395_5_Kvasir.log",'g',0,"ASPP"),
    ],
    [
        ("logs/GCPARCCAResNet_v0/test_GCPARCCAResNet_2021-11-05 15:43:48.718487_5_Kvasir.log",'b',0,"RCCA"),
        ("logs/GCPARCCAResNet_v1/test_GCPARCCAResNet_2021-11-05 15:56:21.709189_5_Kvasir.log",'b',0,"RCCA"),
        ("logs/GCPARCCAResNet_v2/test_GCPARCCAResNet_2021-11-05 15:56:54.234609_5_Kvasir.log",'b',0,"RCCA"),
        ("logs/GCPARCCAResNet_v3/test_GCPARCCAResNet_2021-11-04 21:40:16.255106_5_Kvasir.log",'b',0,"RCCA"),
        ("logs/GCPARCCAResNet_v4/test_GCPARCCAResNet_2021-11-05 16:08:24.508099_5_Kvasir.log",'b',0,"RCCA"),
   ],
     [
        ("logs/GCPACGNLResNet_v0/test_GCPACGNLResNet_2021-11-05 16:08:56.405730_5_Kvasir.log",'y',0,"CGNL"),
        ("logs/GCPACGNLResNet_v1/test_GCPACGNLResNet_2021-11-05 16:15:03.604194_5_Kvasir.log",'y',0,"CGNL"),
        ("logs/GCPACGNLResNet_v2/test_GCPACGNLResNet_2021-11-05 16:16:01.210679_5_Kvasir.log",'y',0,"CGNL"),
        ("logs/GCPACGNLResNet_v3/test_GCPACGNLResNet_2021-11-05 16:19:42.797056_5_Kvasir.log",'y',0,"CGNL"),
        ("logs/GCPACGNLResNet_v3/test_GCPACGNLResNet_2021-11-05 16:19:42.797056_5_Kvasir.log",'y',0,"CGNL"),
   ]
]
path_colors1 = [
    [
        ("logs/GCPAPSPResNet_v0/test_GCPAPSPResNet_2021-11-05 00:42:18.754471_5_CVC-ClinicDB.log",'r',0,"PSP"),
        ("logs/GCPAPSPResNet_v1/test_GCPAPSPResNet_2021-11-05 01:02:29.563087_5_CVC-ClinicDB.log",'r',0,"PSP"),
        ("logs/GCPAPSPResNet_v2/test_GCPAPSPResNet_2021-11-05 01:58:33.324241_5_CVC-ClinicDB.log",'r',0,"PSP"),
        ("logs/GCPAPSPResNet_v3/test_GCPAPSPResNet_2021-11-05 02:19:52.635033_5_CVC-ClinicDB.log",'r',0,"PSP"),
        ("logs/GCPAPSPResNet_v4/test_GCPAPSPResNet_2021-11-05 09:43:00.158861_5_CVC-ClinicDB.log",'r',0,"PSP"),
    ],
    [
        ("logs/GCPAASPPResNet_v0/test_GCPAASPPResNet_2021-11-05 15:50:25.840101_5_CVC-ClinicDB.log",'g',0,"ASPP"),
        ("logs/GCPAASPPResNet_v1/test_GCPAASPPResNet_2021-11-05 10:12:08.994598_5_CVC-ClinicDB.log",'g',0,"ASPP"),
        ("logs/GCPAASPPResNet_v2/test_GCPAASPPResNet_2021-11-05 10:43:07.503579_5_CVC-ClinicDB.log",'g',0,"ASPP"),
        ("logs/GCPAASPPResNet_v3/test_GCPAASPPResNet_2021-11-05 11:08:07.401839_5_CVC-ClinicDB.log",'g',0,"ASPP"),
        ("logs/GCPAASPPResNet_v4/test_GCPAASPPResNet_2021-11-05 15:43:00.054605_5_CVC-ClinicDB.log",'g',0,"ASPP"),
    ],
    [
        ("logs/GCPARCCAResNet_v0/test_GCPARCCAResNet_2021-11-05 15:44:17.740037_5_CVC-ClinicDB.log",'b',0,"RCCA"),
        ("logs/GCPARCCAResNet_v1/test_GCPARCCAResNet_2021-11-05 15:56:30.570172_5_CVC-ClinicDB.log",'b',0,"RCCA"),
        ("logs/GCPARCCAResNet_v2/test_GCPARCCAResNet_2021-11-05 15:57:05.211640_5_CVC-ClinicDB.log",'b',0,"RCCA"),
        ("logs/GCPARCCAResNet_v3/test_GCPARCCAResNet_2021-11-04 21:13:57.247798_5_CVC-ClinicDB.log",'b',0,"RCCA"),
        ("logs/GCPARCCAResNet_v4/test_GCPARCCAResNet_2021-11-05 16:08:30.806237_5_CVC-ClinicDB.log",'b',0,"RCCA"),
   ],
     [
        ("logs/GCPACGNLResNet_v0/test_GCPACGNLResNet_2021-11-05 16:09:05.974203_5_CVC-ClinicDB.log",'y',0,"CGNL"),
        ("logs/GCPACGNLResNet_v1/test_GCPACGNLResNet_2021-11-05 16:15:34.595831_5_CVC-ClinicDB.log",'y',0,"CGNL"),
        ("logs/GCPACGNLResNet_v2/test_GCPACGNLResNet_2021-11-05 16:25:05.267792_5_CVC-ClinicDB.log",'y',0,"CGNL"),
        ("logs/GCPACGNLResNet_v3/test_GCPACGNLResNet_2021-11-05 16:22:13.898076_5_CVC-ClinicDB.log",'y',0,"CGNL"),
        ("logs/GCPACGNLResNet_v3/test_GCPACGNLResNet_2021-11-05 16:22:13.898076_5_CVC-ClinicDB.log",'y',0,"CGNL"),
   ]
]

path_colors =[
    [('logs/GCPA_PSP_orgdataset/test_GCPAPSPNet_2021-03-12 23:10:22.569401_5_Kvasir.log',
   'lime',
   2,
   'PSP'),
  ('logs/GCPA_PSPv2_orgdataset/test_GCPAPSPNet_2021-09-24 11:14:38.747891_5_Kvasir.log',
   'black',
   0,
   'PSPv2'),
  ('logs/GCPAPSPNetv3_orgdataset/test_GCPAPSPNet_2021-09-26 14:32:59.282555_5_Kvasir.log',
   'g',
   0,
   'PSPv3'),
  ('logs/GCPAPSPNetv4_orgdataset/test_GCPAPSPNet_2021-10-03 15:04:08.587240_5_Kvasir.log',
   'g',
   0,
   'PSPv3'),
   
   ], 
      [
    ("logs/GCPA_CC_orgdataset/test_GCPACCNet_2021-03-19 00:22:43.880789_5_Kvasir.log",'cyan',0,"CC"),
    ("logs/GCPARCCA_v1/test_GCPARCCANet_2021-09-29 02:55:18.818553_5_Kvasir.log",'cyan',0,"CC"),
    ("logs/GCPARCCA_v2/test_GCPARCCANet_2021-09-30 00:36:16.335798_5_Kvasir.log",'cyan',0,"CC"),
    ("logs/GCPARCCA_v3/test_GCPARCCANet_2021-09-30 01:01:03.695057_5_Kvasir.log",'cyan',0,"CC"),
    ("logs/GCPARCCA_v4/test_GCPARCCANet_2021-10-03 15:18:31.274342_5_Kvasir.log",'cyan',0,"CC"),
   ],
   [
    ("logs/GCPA_ASPP_orgdataset/test_GCPAASPPNet_2021-03-12 06:53:41.989173_5_Kvasir.log",'m',2,"ASPP"),
    ("logs/GCPAASPP_v1/test_GCPAASPPNet_2021-09-30 20:02:43.242346_5_Kvasir.log",'m',0,"ASPP"),
    ("logs/GCPAASPP_v2/test_GCPAASPPNet_2021-10-01 21:38:39.123464_5_Kvasir.log",'m',0,"ASPP"),
    ("logs/GCPAASPP_v3/test_GCPAASPPNet_2021-10-01 21:39:39.738335_5_Kvasir.log",'m',0,"ASPP"),
    ("logs/GCPAASPP_v4/test_GCPAASPPNet_2021-10-01 21:43:38.008801_5_Kvasir.log",'m',0,"ASPP"),
   ],

#    [
#     ("logs/SCWSRCCA_orgdataset/test_SCWSRCCANet_2021-05-08 23:30:36.764196_5_Kvasir.log",'b',0,"FAMv2-CC"),
#     ("logs/SCWSRCCAv2_orgdataset/test_SCWSRCCANet_2021-09-21 20:06:34.104297_5_Kvasir.log",'b',0,"FAMv2-CC"),
#     ("logs/SCWSRCCAv3_orgdataset/test_SCWSRCCANet_2021-09-26 15:52:19.492478_5_Kvasir.log",'b',0,"FAMv2-CC"),
#     ("logs/SCWSRCCAv4_orgdataset/test_SCWSRCCANet_2021-09-26 14:34:23.656733_5_Kvasir.log",'b',0,"FAMv2-CC"),
#     ("logs/SCWSRCCAv5_orgdataset/test_SCWSRCCANet_2021-09-27 21:55:29.182838_5_Kvasir.log",'b',0,"FAMv2-CC"),
#    ],
      [
    ("logs/GCPA_GALDv8_orgdataset/test_GCPAGALDNetv8_2021-03-10 10:58:50.415639_5_Kvasir.log",'r',2,"CGNL"),
    ("logs/GCPAGALDNetv8_kfoldv1/test_GCPAGALDNetv8_2021-09-29 01:18:57.137598_5_Kvasir.log",'r',0,"CGNL"),
    ("logs/GCPAGALDNetv8_kfoldv2/test_GCPAGALDNetv8_2021-09-29 01:20:17.857876_5_Kvasir.log",'r',0,"CGNL"),
    ("logs/GCPAGALDNetv8_kfoldv3/test_GCPAGALDNetv8_2021-09-29 01:42:52.984295_5_Kvasir.log",'r',0,"CGNL"),
    ("logs/GCPAGALDNetv8_kfoldv4/test_GCPAGALDNetv8_2021-09-29 01:47:01.257980_5_Kvasir.log",'r',0,"CGNL"),
   ]
   
   ]

path_colors1 =[
    [('logs/GCPA_PSP_orgdataset/test_GCPAPSPNet_2021-03-12 23:10:44.953094_5_CVC-ClinicDB.log',
   'lime',
   2,
   'PSP'),
  ('logs/GCPA_PSPv2_orgdataset/test_GCPAPSPNet_2021-09-23 20:11:55.415501_5_CVC-ClinicDB.log',
   'black',
   0,
   'PSPv2'),
  ('logs/GCPAPSPNetv3_orgdataset/test_GCPAPSPNet_2021-09-26 14:32:34.727353_5_CVC-ClinicDB.log',
   'g',
   0,
   'PSPv3'),
   ('logs/GCPAPSPNetv4_orgdataset/test_GCPAPSPNet_2021-10-03 15:14:33.246647_5_CVC-ClinicDB.log',
   'g',
   0,
   'PSPv3')
   ],
   [
    ("logs/GCPA_CC_orgdataset/test_GCPACCNet_2021-03-19 00:25:33.623405_5_CVC-ClinicDB.log",'cyan',0,"CC"),
    ("logs/GCPARCCA_v1/test_GCPARCCANet_2021-09-29 02:54:35.574069_5_CVC-ClinicDB.log",'cyan',0,"CC"),
    ("logs/GCPARCCA_v2/test_GCPARCCANet_2021-09-30 00:36:49.379358_5_CVC-ClinicDB.log",'cyan',0,"CC"),
    ("logs/GCPARCCA_v3/test_GCPARCCANet_2021-09-30 01:00:44.437433_5_CVC-ClinicDB.log",'cyan',0,"CC"),
    ("logs/GCPARCCA_v4/test_GCPARCCANet_2021-10-03 15:18:20.188562_5_CVC-ClinicDB.log",'cyan',0,"CC"),
   ],
    [
    ("logs/GCPA_ASPP_orgdataset/test_GCPAASPPNet_2021-03-12 06:57:14.967380_5_CVC-ClinicDB.log",'m',2,"ASPP"),
    ("logs/GCPAASPP_v1/test_GCPAASPPNet_2021-09-30 20:03:19.788926_5_CVC-ClinicDB.log",'m',0,"ASPP"),
    ("logs/GCPAASPP_v2/test_GCPAASPPNet_2021-10-01 21:38:26.941264_5_CVC-ClinicDB.log",'m',0,"ASPP"),
    ("logs/GCPAASPP_v3/test_GCPAASPPNet_2021-10-01 21:39:59.752319_5_CVC-ClinicDB.log",'m',0,"ASPP"),
    ("logs/GCPAASPP_v4/test_GCPAASPPNet_2021-10-01 21:52:43.679225_5_CVC-ClinicDB.log",'m',0,"ASPP"),
   ],
#    [
#     ("logs/SCWSRCCA_orgdataset/test_SCWSRCCANet_2021-05-08 23:35:17.888119_5_CVC-ClinicDB.log",'b',0,"FAMv2-CC"),
#     ("logs/SCWSRCCAv2_orgdataset/test_SCWSRCCANet_2021-09-21 20:06:15.295115_5_CVC-ClinicDB.log",'b',0,"FAMv2-CC"),
#     ("logs/SCWSRCCAv3_orgdataset/test_SCWSRCCANet_2021-09-26 00:17:08.579013_5_CVC-ClinicDB.log",'b',0,"FAMv2-CC"),
#     ("logs/SCWSRCCAv4_orgdataset/test_SCWSRCCANet_2021-09-26 14:34:44.322734_5_CVC-ClinicDB.log",'b',0,"FAMv2-CC"),
#     ("logs/SCWSRCCAv5_orgdataset/test_SCWSRCCANet_2021-09-27 21:56:11.610889_5_CVC-ClinicDB.log",'b',0,"FAMv2-CC"),
#    ],

      [
    ("logs/GCPA_GALDv8_orgdataset/test_GCPAGALDNetv8_2021-03-10 11:02:34.064225_5_CVC-ClinicDB.log",'r',2,"CGNL"),
    ("logs/GCPAGALDNetv8_kfoldv1/test_GCPAGALDNetv8_2021-09-29 01:36:50.749205_5_CVC-ClinicDB.log",'r',0,"CGNL"),
    ("logs/GCPAGALDNetv8_kfoldv2/test_GCPAGALDNetv8_2021-09-29 01:30:35.639106_5_CVC-ClinicDB.log",'r',0,"CGNL"),
    ("logs/GCPAGALDNetv8_kfoldv3/test_GCPAGALDNetv8_2021-09-29 01:42:23.383489_5_CVC-ClinicDB.log",'r',0,"CGNL"),
    ("logs/GCPAGALDNetv8_kfoldv4/test_GCPAGALDNetv8_2021-09-29 01:53:12.439749_5_CVC-ClinicDB.log",'r',0,"CGNL"),
   ]
   
   ]

from scipy.ndimage.filters import gaussian_filter1d
sigma = 1
fig, axs = plt.subplots(2, 1,constrained_layout=True)
# fig.tight_layout()
# 0 - new
# 1 - old 
# 2 - new but fake
# o = []
for path_color in path_colors:
    o = []
    for path, color, ty, label in path_color:
        data = (open(path,"r").read().split("\n"))
        if ty == 0:
            x = [int(data[i][-3:]) for i in range(2, len(data), 6) if len(data[i][-3:]) > 1]
            o.append([(int(data[i][-3:])) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3])
        elif ty == 1:
            x = [int(data[i][-3:]) for i in range(2, len(data), 5) if len(data[i][-3:]) > 1]
            o.append([(int(data[i][-3:])) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3])
        elif ty == 2:
            x = [int(data[i][-3:])//3+134 for i in range(2, len(data), 6) if len(data[i][-3:]) > 1]
            o.append([(int(data[i][-3:])) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3])
        elif ty == 3:
            x = [int(data[i][-3:])//3+134 for i in range(2, len(data), 5) if len(data[i][-3:]) > 1]
            o.append([(int(data[i][-3:])) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3])
        elif ty == 4:
            x = [int(data[i][-3:])//3+140 for i in range(2, len(data), 5) if len(data[i][-3:]) > 1]
            o.append([(int(data[i][-3:])) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3])

    o = np.array([i[-40:] for i in o])
    train_scores_mean = np.mean(o, axis=0)
    train_scores_std = np.std(o, axis=0)

    if("v2" in path_color[0][3]):
        axs[0].fill_between(x[-40:], gaussian_filter1d(train_scores_mean - train_scores_std,sigma=sigma),
                        gaussian_filter1d(train_scores_mean + train_scores_std,sigma=sigma), alpha=0.1,
                        color=path_color[0][1],linestyle='dashed')
        axs[0].plot(x[-40:], gaussian_filter1d(train_scores_mean, sigma=sigma), color=path_color[0][1],
                    label=path_color[0][3],linestyle='dashed')
        # axs[0].plot(x[-40:], gaussian_filter1d(y[-40:], sigma=sigma),label = path_color[0][3], color=path_color[0][1],linestyle='dashed')
    else:
        if (sigma == 0):
            axs[0].fill_between(x[-40:], train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color=path_color[0][1])
            axs[0].plot(x[-40:] , train_scores_mean, color=path_color[0][1],
                        label=path_color[0][3])

            # axs[0].plot(x[-40:], y[-40:],label = path_color[0][3], color=path_color[0][1])
        else:
            axs[0].fill_between(x[-40:], gaussian_filter1d(train_scores_mean - train_scores_std, sigma=sigma),
                gaussian_filter1d(train_scores_mean + train_scores_std, sigma=sigma), alpha=0.1,
                color=path_color[0][1])
            axs[0].plot( x[-40:] , gaussian_filter1d(train_scores_mean,sigma=sigma), color=path_color[0][1],
                        label=path_color[0][3])

            # axs[0].plot(x[-40:], gaussian_filter1d(y[-40:], sigma=sigma),label = path_color[0][3], color=path_color[0][1])
    axs[0].set_title("Kvasir")
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("DSC(%)") 
    axs[0].legend(bbox_to_anchor =(1, 1.6), ncol = 4)
    # axs[0].legend(loc='upper right', frameon=False)



for path_color in path_colors1:
    o = []
    for path, color, ty, label in path_color:
        data = (open(path,"r").read().split("\n"))
        if ty == 0:
            x = [int(data[i][-3:]) for i in range(2, len(data), 6) if len(data[i][-3:]) > 1]
            o.append([(int(data[i][-3:])) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3])
        elif ty == 1:
            x = [int(data[i][-3:]) for i in range(2, len(data), 5) if len(data[i][-3:]) > 1]
            o.append([(int(data[i][-3:])) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3])
        elif ty == 2:
            x = [int(data[i][-3:])//3+134 for i in range(2, len(data), 6) if len(data[i][-3:]) > 1]
            o.append([(int(data[i][-3:])) for i in range(4, len(data), 6) if len(data[i][-3:]) == 3])
        elif ty == 3:
            x = [int(data[i][-3:])//3+134 for i in range(2, len(data), 5) if len(data[i][-3:]) > 1]
            o.append([(int(data[i][-3:])) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3])
        elif ty == 4:
            x = [int(data[i][-3:])//3+140 for i in range(2, len(data), 5) if len(data[i][-3:]) > 1]
            o.append([(int(data[i][-3:])) for i in range(3, len(data), 5) if len(data[i][-3:]) == 3])

    o = np.array([i[-40:] for i in o])
    train_scores_mean = np.mean(o, axis=0)
    train_scores_std = np.std(o, axis=0)

    if("v2" in path_color[0][3]):
        axs[1].fill_between(x[-40:], gaussian_filter1d(train_scores_mean - train_scores_std,sigma=sigma),
                        gaussian_filter1d(train_scores_mean + train_scores_std, sigma=sigma), alpha=0.1,
                        color=path_color[0][1],linestyle='dashed')
        axs[1].plot(x[-40:], gaussian_filter1d(train_scores_mean, sigma=sigma), color=path_color[0][1],
                    label=path_color[0][3],linestyle='dashed')
        # axs[0].plot(x[-40:], gaussian_filter1d(y[-40:], sigma=sigma),label = path_color[0][3], color=path_color[0][1],linestyle='dashed')
    else:
        if (sigma == 0):
            axs[1].fill_between(x[-40:], train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color=path_color[0][1])
            axs[1].plot(x[-40:] , train_scores_mean, color=path_color[0][1],
                        label=path_color[0][3])

            # axs[0].plot(x[-40:], y[-40:],label = path_color[0][3], color=path_color[0][1])
        else:
            axs[1].fill_between(x[-40:], gaussian_filter1d(train_scores_mean - train_scores_std, sigma=sigma),
                gaussian_filter1d(train_scores_mean + train_scores_std, sigma=sigma), alpha=0.1,
                color=path_color[0][1])
            axs[1].plot( x[-40:] , gaussian_filter1d(train_scores_mean,sigma=sigma), color=path_color[0][1],
                        label=path_color[0][3])

    axs[1].set_title("Clinic")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("DSC(%)")
    # axs[1].legend(bbox_to_anchor =(0.75, 1.15), ncol = 2)
    # axs[1].legend(loc='best')

o = np.array([i[-40:] for i in o])
v = np.append([x[-40:]],o, axis=0)
fileHeader =  ["epoch", "PSPv0", "PSPv2", "PSPv3","PSPv4","CCv0","CCv1","CCv2","CCv3","CCv4","ASPPv0","ASPPv1","ASPPv2","ASPPv3","ASPPv4","CGNLv0","CGNLv1","CGNLv2","CGNLv3","CGNLv4"]
import pandas as pd
df = pd.DataFrame(v.T)
df.columns = fileHeader
df.to_csv('clinic.csv',index=False)
