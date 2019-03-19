import os


def getPid(process):
    # adb shell "ps -ef|grep "procename""
    cmd = 'adb shell ' + '"' + 'ps -ef|grep ' + '"' + process + '"' + '"'
    print(cmd)


def push_minicap(path=None):
    '''
    :param path: minicap路径
    :return:
    '''
    cpu_cmd = "adb shell getprop ro.product.cpu.abi"
    sdk_cmd = "adb shell getprop ro.build.version.sdk"
    cpu_info = os.popen(cpu_cmd).read().strip('\n')
    sdk_info = os.popen(sdk_cmd).read().strip('\n')
    if cpu_info is None or sdk_info is None:
        print("get cpuinfo or sdk ino error")
    else:
        push_lib = 'adb push ' + os.path.join(path, 'libs', cpu_info, 'minicap') + ' /data/local/tmp/'
        print(push_lib)
        push_jni = 'adb push ' + os.path.join(path, 'jni\minicap-shared\\aosp\libs', "android-" + sdk_info, cpu_info,
                                              'minicap.so') + ' /data/local/tmp/'
        print(push_jni)
        os.popen(push_lib)
        os.popen(push_jni)
        cmd = 'adb shell chmod 0755 /data/local/tmp/minicap'
        os.popen(cmd)


def push_minitouch(path=None):
    '''
    :param path: minitouch路径
    :return:
    '''
    cpu_cmd = "adb shell getprop ro.product.cpu.abi"
    sdk_cmd = "adb shell getprop ro.build.version.sdk"
    cpu_info = os.popen(cpu_cmd).read().strip('\n')
    sdk_info = os.popen(sdk_cmd).read().strip('\n')
    if cpu_info is None or sdk_info is None:
        print("get cpuinfo or sdk ino error")
    else:
        push_lib = 'adb push ' + os.path.join(path, 'libs', cpu_info, 'minitouch') + ' /data/local/tmp/'
        print(push_lib)
        os.popen(push_lib)
        cmd = 'adb shell chmod 0755 /data/local/tmp/minitouch'
        os.popen(cmd)


def start_minitouch():
    cmd = "adb shell /data/local/tmp/minitouch"
    print(cmd)
    os.popen(cmd)

def get_width_height():
    '''

    :return:返回屏幕宽和高
    '''
    cmd = 'adb shell wm size'
    output = os.popen(cmd).read().strip('\n').split(' ')  # ['Physical', 'size:', '720x1280']
    width = output[2].split("x")[0]
    height = output[2].split("x")[1]
    return width, height


if __name__ == "__main__":
    minicap_path = r"C:\Users\moongong\minicap"
    minitouch_path = r"C:\Users\moongong\minitouch"
    push_minitouch(minitouch_path)
    push_minicap(minicap_path)



