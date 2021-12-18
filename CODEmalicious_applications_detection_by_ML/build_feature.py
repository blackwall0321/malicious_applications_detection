import pandas as pd
import xml.etree.ElementTree as ET
import os,sys
import json
import codecs


def create_empty_table(csv_path):
    df = pd.DataFrame(columns=['app_name'])
    df.to_csv(csv_path, index=False)
def insert_column(csv_path,xml_path):
    #辩论问价
    df = pd.read_table(csv_path)
    data = df.values
    num = len(data)
    for root, dirs, files in os.walk(xml_path):
        for file in files:
            print(file)
            df.loc[num,'app_name']=file.split('.')[0]
            num = num+ 1
    df.to_csv(csv_path,index=False)

def permission_level(permission_name):
    num = 1

    level1 = ['ACCESS_CHECKIN_PROPERTIES','ACCOUNT_MANAGER','BATTERY_STATS','BIND_APPWIDGET','BIND_CARRIER_MESSAGING_SERVICE','BIND_QUICK_SETTINGS_TILE','BIND_REMOTEVIEWS','BLUETOOTH_PRIVILEGED','BROADCAST_PACKAGE_REMOVED','BROADCAST_SMS','BROADCAST_WAP_PUSH','CALL_PRIVILEGED','CAPTURE_AUDIO_OUTPUT','CAPTURE_SECURE_VIDEO_OUTPUT','CAPTURE_VIDEO_OUTPUT','CHANGE_COMPONENT_ENABLED_STATE','CHANGE_CONFIGURATION','CONTROL_LOCATION_UPDATES','DELETE_CACHE_FILES','DELETE_PACKAGES','DIAGNOSTIC','DUMP','FACTORY_TEST','GET_ACCOUNTS_PRIVILEGED','GET_TASKS','GLOBAL_SEARCH','INSTALL_LOCATION_PROVIDER','INSTALL_PACKAGES','INSTANT_APP_FOREGROUND_SERVICE','LOCATION_HARDWARE','MASTER_CLEAR','MEDIA_CONTENT_CONTROL','MODIFY_PHONE_STATE','MOUNT_FORMAT_FILESYSTEMS','MOUNT_UNMOUNT_FILESYSTEMS','PACKAGE_USAGE_STATS','PERSISTENT_ACTIVITY','READ_CALL_LOG','READ_FRAME_BUFFER','READ_INPUT_STATE','READ_LOGS','REBOOT','REQUEST_IGNORE_BATTERY_OPTIMIZATIONS','RESTART_PACKAGES','SEND_RESPOND_VIA_MESSAGE','SET_ALWAYS_FINISH','SET_ANIMATION_SCALE','SET_DEBUG_APP','SET_PREFERRED_APPLICATIONS','SET_PROCESS_LIMIT','SET_TIME','SET_TIME_ZONE','SIGNAL_PERSISTENT_PROCESSES','STATUS_BAR','UNINSTALL_SHORTCUT','UPDATE_DEVICE_STATS','WAKE_LOCK','WRITE_APN_SETTINGS','WRITE_GSERVICES','WRITE_SECURE_SETTINGS']
    level2 = ['ACCESS_LOCATION_EXTRA_COMMANDS','ACCESS_NETWORK_STATE','ACCESS_NOTIFICATION_POLICY','ACCESS_WIFI_STATE','BLUETOOTH','BLUETOOTH_ADMIN','BROADCAST_STICKY','CHANGE_NETWORK_STATE','CHANGE_WIFI_MULTICAST_STATE','CHANGE_WIFI_STATE','DISABLE_KEYGUARD','EXPAND_STATUS_BAR','FOREGROUND_SERVICE','GET_PACKAGE_SIZE','INSTALL_SHORTCUT','INTERNET','KILL_BACKGROUND_PROCESSES','MANAGE_OWN_CALLS','MODIFY_AUDIO_SETTINGS','NFC','NFC_TRANSACTION_EVENT','READ_SMS','READ_SYNC_SETTINGS','READ_SYNC_STATS','RECEIVE_BOOT_COMPLETED','REORDER_TASKS','REQUEST_COMPANION_RUN_IN_BACKGROUND','REQUEST_COMPANION_USE_DATA_IN_BACKGROUND','REQUEST_DELETE_PACKAGES','SET_ALARM','SET_WALLPAPER','SET_WALLPAPER_HINTS','TRANSMIT_IR','USE_BIOMETRIC','USE_FINGERPRINT','VIBRATE','WRITE_SYNC_SETTINGS']
    level3 = ['ACCEPT_HANDOVER','ACCESS_COARSE_LOCATION','ACCESS_FINE_LOCATION','ADD_VOICEMAIL','ANSWER_PHONE_CALLS','BODY_SENSORS','CALL_PHONE','CAMERA','GET_ACCOUNTS','PROCESS_OUTGOING_CALLS','READ_CALENDAR','READ_CONTACTS','READ_EXTERNAL_STORAG','READ_PHONE_NUMBERS','READ_PHONE_STATE','RECEIVE_MMS','RECEIVE_SMS','RECEIVE_WAP_PUSH','RECORD_AUDIO','SEND_SMS','USE_SIP','WRITE_CALENDAR','WRITE_CALL_LOG','WRITE_CONTACTS','WRITE_EXTERNAL_STORAGE']
    level4 = ['BIND_ACCESSIBILITY_SERVICE','BIND_AUTOFILL_SERVICE','BIND_CHOOSER_TARGET_SERVICE','BIND_CONDITION_PROVIDER_SERVICE','BIND_DEVICE_ADMIN','BIND_DREAM_SERVICE','BIND_INPUT_METHOD','BIND_MIDI_DEVICE_SERVICE','BIND_NFC_SERVICE','BIND_NOTIFICATION_LISTENER_SERVICE','BIND_PRINT_SERVICE','BIND_TEXT_SERVICE','BIND_VOICE_INTERACTION','BIND_VPN_SERVICE','BIND_VR_LISTENER_SERVICE','MANAGE_DOCUMENTS','REQUEST_INSTALL_PACKAGES','SYSTEM_ALERT_WINDOW','WRITE_SETTINGS']
    level5 = ['BIND_CARRIER_SERVICES','BIND_INCALL_SERVICE','BIND_SCREENING_SERVICE','BIND_TELECOM_CONNECTION_SERVICE','BIND_TV_INPUT','BIND_VISUAL_VOICEMAIL_SERVICE','BIND_WALLPAPER','CLEAR_APP_CACHE','READ_VOICEMAIL','WRITE_VOICEMAIL']
    if permission_name in level1:
        num = 2
    if permission_name in level2:
        num = 3
    if permission_name in level3:
        num = 4
    if permission_name in level4:
        num = 5
    if permission_name in level5:
        num = 6
    return num

def xml_feature(xml_path,csv_path,flags):
    permission = []
    intent = []
    uses_feature =[]
    df = pd.read_csv(csv_path,sep = ',',header=0,index_col='app_name')
    for root, dirs, files in os.walk(xml_path):
        for xml_file in files:
            #print(xml_file)
            try:
                tree = ET.parse(xml_path + '/' + xml_file)  # 解析xml文件
                cache = tree.getroot()  # 获取xml的根节点
                for child in cache:
                    if child.tag == 'uses-permission' and child.attrib['{http://schemas.android.com/apk/res/android}name'][0:18] == 'android.permission':
                        permission_name = child.attrib['{http://schemas.android.com/apk/res/android}name'].split('permission.')[1]
                        df.loc[xml_file.split('.')[0],permission_name]  = permission_level(permission_name)
                            #permission_level(permission_name)
                        permission.append(permission_name)
                    if child.tag == 'uses-feature' and child.attrib['{http://schemas.android.com/apk/res/android}name'][0:8] == 'android.':
                        uses_feature_name =child.attrib['{http://schemas.android.com/apk/res/android}name'][8:]
                        df.loc[xml_file.split('.')[0],  uses_feature_name] = 1
                        uses_feature.append(child.attrib['{http://schemas.android.com/apk/res/android}name'].split('android.')[1])
                    for er in child:
                        for san in er:
                            for si in san:
                                for android_name in si.attrib:
                                    if android_name == '{http://schemas.android.com/apk/res/android}name' and si.attrib[android_name][0:15] == 'android.intent.':
                                        intent_name = si.attrib[android_name][15:]
                                        df.loc[xml_file.split('.')[0], intent_name] = 1
                                        intent.append(intent_name)
            except:
                df.loc[xml_file.split('.')[0], 'safe_or_bad'] = flags
                continue
    intent = list(set(intent))
    print(len(intent))
    permission = list(set(permission))
    print(len(permission))
    uses_feature = list(set(uses_feature))
    print(len(uses_feature))
    df.to_csv(csv_path,index=True)

def androguard_feature(txt_path,csv_path):
    df = pd.read_csv(csv_path, sep=',', header=0, index_col='app_name')
    txt_num = 0
    for root, dirs, files in os.walk(txt_path):
        for txt in files:
            print(txt,txt_num)
            txt_num = txt_num+1
            try:
                file_num  = 0
                Activity_num = 1
                Receiver_num = 0
                Service_num = 0
                permission_num = 0
                fread = open(root + '\\' + txt,encoding='gbk')
                str = fread.readline()
                str = fread.readline()
                if str[0:7] == 'INVALID' or str[0:5] =='ERROR':
                    continue
                flag1 = flag2 = flag3 = flag4 = 0
                bug = False
                if str[0:6]=='FILES:':
                    while(str[0:12]!='PERMISSIONS:'):
                        if str[1:6] !='ERROR':
                            file_num = file_num + 1
                            str = fread.readline()
                            #print(str[0:10])
                        else:
                            bug = True
                            break
                if bug:
                    continue
                if str[0:12]=='PERMISSIONS:':
                    while (str[0:14] != 'MAIN ACTIVITY:'):
                        permission_num = permission_num + 1
                        str = fread.readline()
                str = fread.readline()
                if str[0:11] == 'ACTIVITIES:':
                    while (str[0:9] != 'SERVICES:'):
                        if str[0:5] !='ERROR':
                            Activity_num = Activity_num + 1
                            str = fread.readline()
                        else:
                            bug = True
                            break
                if bug:
                    continue
                if str[0:9] == 'SERVICES:':
                    while (str[0:10] != 'RECEIVERS:'):
                        Service_num = Service_num + 1
                        str = fread.readline()
                if str[0:10] == 'RECEIVERS:':
                    while (str[0:10] != 'PROVIDERS:'):
                        Receiver_num =   Receiver_num + 1
                        str = fread.readline()
                str = fread.readline()
                if str[0:5] == 'ERROR':
                    continue

                while(str[0:11]!='Native code'):
                    if str:
                        str = fread.readline()
                    else:
                        continue

                if str[0:11] == 'Native code':
                    str1 = fread.readline()
                    str2 = fread.readline()
                    str3 = fread.readline()
                if (str[13:].split(' ')[0][0:4] == 'True'):
                    flag1 = 1
                else:
                    #print(str[13:].split(' ')[0][0:4])
                    flag1 = 0

                if (str1[14:].split(' ')[0][0:4] == 'True'):
                    flag2 = 1
                else:
                    flag2 = 0

                if (str2[17:].split(' ')[0][0:4] == 'True'):
                    flag3 = 1
                else:
                    flag3 = 0

                if (str3[19:].split(' ')[0][0:4] == 'True'):
                    flag4 = 1
                else:
                    if str3[19:].split(' ')[0][0:5] == 'ERROR':
                        flag4 = 0
                    else:
                        flag4 = -1
            except:
                print( '------------------>',txt)

            filename = txt.split('.')[0]
            df.loc[filename, 'Native_code'] = flag1
            df.loc[filename, 'Dynamic_code'] = flag2
            df.loc[filename, 'Reflection_code'] = flag3
            df.loc[filename, 'Ascii_Obfuscation'] = flag4
            df.loc[filename, 'Activity_num'] = Activity_num
            df.loc[filename, 'Service_num'] = Service_num
            df.loc[filename, 'Receiver_num'] = Receiver_num
            df.loc[filename, 'permission_num'] = permission_num
            df.loc[filename, 'file_num'] = file_num


    df.to_csv(csv_path, index=True)

def droidbox(droid_txt_path,csv_path):
    df = pd.read_csv(csv_path, sep=',', header=0, index_col='app_name')
    for index, dic in enumerate(os.listdir(droid_txt_path)):
        try:
            with codecs.open(os.path.join(droid_txt_path, dic), 'r', encoding='utf-8') as f:
                data = f.readlines()
                data = data[-1]
                data = json.loads(data.strip('\n'))

                if isinstance(data, dict):
                    apk_name = data['apkName']
                    app_name = dic.split('.')[0]

                    enfperm = data['enfperm']
                    df.loc[app_name, 'enfperm_num'] = len(enfperm)

                    recvnet = data['recvnet']
                    df.loc[app_name, 'recvnet_num'] = len(recvnet)

                    servicestart = data['servicestart']
                    df.loc[app_name, 'servicestart_num'] = len(servicestart)

                    sendsms = data['sendsms']
                    df.loc[app_name, 'sendsms_num'] = len(sendsms)

                    cryptousage = data['cryptousage']
                    encryption_num = keyalgo_num = decryption_num = 0
                    if isinstance(cryptousage, dict):
                        df.loc[dic.split('.')[0], 'cryptousage_num'] = len(cryptousage)
                        algorithm = ['DESede/CBC/PKCS5Padding', 'Blowfish', 'HmacSHA1', 'DESede',
                                     'desede/ECB/PKCS5Padding', 'AES',
                                     'PBEWithMD5AndDES', 'AES/ECB/PKCS7Padding', 'RSA', 'Blowfish/CBC/NoPadding',
                                     'AES/CBC/PKCS5Padding', 'DES', 'RSA/ECB/PKCS1Padding',
                                     'desede/CBC/PKCS5Padding', 'PBEWITHMD5andDES', 'AES/ECB/PKCS5Padding',
                                     'DES/ECB/PKCS5Padding', 'AES/CBC/NoPadding', 'DES/CBC/PKCS5Padding']
                        algorithm_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]

                        for key1 in cryptousage:
                            if cryptousage[key1]['operation'] == 'encryption':
                                encryption_num += 1
                            if cryptousage[key1]['operation'] == 'keyalgo':
                                keyalgo_num += 1
                            if cryptousage[key1]['operation'] == 'decryption':
                                decryption_num += 1
                            if cryptousage[key1]['algorithm'] in algorithm:
                                algorithm_num[algorithm.index(cryptousage[key1]['algorithm'])] += 1

                        df.loc[app_name, 'cryptousage_operation_encryption_num'] = encryption_num
                        df.loc[app_name, 'cryptousage_operation_keyalgo_num'] = keyalgo_num
                        df.loc[app_name, 'cryptousage_operation_decryption_num'] = decryption_num
                        for i in range(19):
                            df.loc[app_name, 'algorithm_' + algorithm[i]] = algorithm_num[i]

                    sendnet = data['sendnet']
                    sendnet_tag = ['TAINT_LOCATION', 'TAINT_ICCID', 'TAINT_IMEI', 'TAINT_PHONE_NUMBER',
                                   'TAINT_LOCATION_GPS', 'TAINT_IMSI']
                    sendnet_tag_num = [0, 0, 0, 0, 0, 0]
                    if isinstance(sendnet, dict):
                        df.loc[app_name, 'sendney_num'] = len(sendnet)
                        sink_num = 0

                        # print(app_name)
                        for key in sendnet:
                            for i in sendnet[key]:
                                if i == 'sink':
                                    sink_num += 1
                                if i == 'tag':
                                    for j in sendnet[key]['tag']:
                                        if j in sendnet_tag:
                                            sendnet_tag_num[sendnet_tag.index(j)] += 1

                        df.loc[app_name, 'sink_num'] = sink_num
                        for j in range(6):
                            df.loc[app_name, 'sendnet_tag_' + sendnet_tag[j]+'_num'] = sendnet_tag_num[j]
                    accessedfiles = data['accessedfiles']
                    if isinstance(accessedfiles, dict):
                        df.loc[app_name, 'accessedfiles'] = len(accessedfiles)
                    fdaccess = data['fdaccess']
                    if isinstance(fdaccess, dict):
                        df.loc[app_name, 'fdaccess'] = len(fdaccess)
                        write = read = 0
                        for key in fdaccess:
                            if fdaccess[key]['operation'] == 'write':
                                write = write + 1
                            if fdaccess[key]['operation'] == 'read':
                                read = read + 1
                        df.loc[app_name, 'fdaccess_operation_write'] = write
                        df.loc[app_name, 'fdaccess_operation_read'] = read
                    dataleaks = data['dataleaks']
                    write = read = 0
                    dataleaks_sink_File = dataleaks_sink_Network = 0
                    if isinstance(dataleaks, dict):
                        for key in dataleaks:
                            if dataleaks[key]['operation'] == 'write':
                                write = write + 1
                            if dataleaks[key]['operation'] == 'read':
                                read = read + 1
                            if dataleaks[key]['sink'] == 'File':
                                dataleaks_sink_File += 1
                            if dataleaks[key]['sink'] == 'Network':
                                dataleaks_sink_Network += 1
                        df.loc[app_name, 'dataleaks_operation_write'] = write
                        df.loc[app_name, 'dataleaks_operation_read'] = read
                        df.loc[app_name, 'dataleaks_sink_File'] = dataleaks_sink_File
                        df.loc[app_name, 'dataleaks_sink_Network'] = dataleaks_sink_Network
                    opennet = data['opennet']
                    if isinstance(opennet, dict):
                        df.loc[app_name, 'opennet'] = len(opennet)
                    recvsaction = data['recvsaction']
                    if isinstance(recvsaction, dict):
                        df.loc[app_name, 'recvsaction'] = len(recvsaction)

                    dexclass = data['dexclass']

                    hashes = data['hashes']

                    closenet = data['closenet']

                    phonecalls = data['phonecalls']
        except:
            pass
    df.to_csv(csv_path, index=True)



csv_path = 'E:\Droid-LMAD\source_code\\table\\empty.csv'

create_empty_table(csv_path)
good_xml_path = 'J:\实验数据\\final_good_4_temp_xml'
bad_xml_path ='J:\实验数据\\final_bad_1_temp_xml'
insert_column(csv_path,good_xml_path)
insert_column(csv_path,bad_xml_path)


xml_feature(good_xml_path,csv_path,1)
good_txt_path='J:\实验数据\\final_good_4_androguard'
androguard_feature(good_txt_path,csv_path)

xml_feature(bad_xml_path,csv_path,0)
bad_txt_path='J:\实验数据\\final_bad_1_androguard'
androguard_feature(bad_txt_path,csv_path)



droid_txt_path = 'J:\实验数据\\final_bad_1_droidbox'
droidbox(droid_txt_path,csv_path )
droid_txt_path = 'J:\实验数据\\final_good_4_droidbox'
droidbox(droid_txt_path,csv_path )
