# encoding=utf-8

import os
import math
import logging

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ==============================
# 全局配置
# ==============================

# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 支持的图片格式
SUPPORTED_FORMATS = ('.bmp', '.png')

# 最大允许秘密信息长度（字符）
MAX_MESSAGE_LENGTH = 1024

# DCT保留低频区域大小
LOW_FREQUENCY_SIZE = 300

# 日志配置
logging.basicConfig(
    filename="lsb.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ==============================
# 安全检查函数
# ==============================

def validate_message(msg):
    """
    检查秘密信息是否合法
    """

    if not isinstance(msg, str):
        raise TypeError("秘密信息必须为字符串！")

    if len(msg.strip()) == 0:
        raise ValueError("秘密信息不能为空！")

    if len(msg) > MAX_MESSAGE_LENGTH:
        raise ValueError(
            f"秘密信息长度超过限制({MAX_MESSAGE_LENGTH}字符)"
        )


def validate_image(path):
    """
    检查图片是否合法
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(f"图片不存在：{path}")

    suffix = os.path.splitext(path)[1].lower()

    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"仅支持以下图片格式：{SUPPORTED_FORMATS}"
        )


def check_capacity(img_path, msg):
    """
    检查图片容量是否足够
    """

    img = Image.open(img_path)

    width, height = img.size

    # RGB三个通道，每通道隐藏1bit
    capacity = width * height * 3

    bits = len(get_binary_msg(msg))

    if bits > capacity:
        raise ValueError(
            f"秘密信息长度({bits} bits)超过图片最大容量({capacity} bits)"
        )

    logging.info("容量检查通过")


# ==============================
# 基础工具函数
# ==============================

def get_binary_msg(msg):
    """
    将字符串转换为二进制字符串
    """

    validate_message(msg)

    binary = ''.join(
        format(byte, '08b')
        for byte in msg.encode("utf-8")
    )

    return binary


def psnr(img1, img2):
    """
    计算PSNR
    """

    img1 = np.float64(img1)
    img2 = np.float64(img2)

    mse = np.mean((img1 - img2) ** 2)

    if mse < 1e-10:
        return 100

    return 20 * math.log10(
        255.0 / math.sqrt(mse)
    )

# ==============================
# 图像预处理
# ==============================

def preprocess_image(input_path, output_path):
    """
    图像灰度化并缩放到512×512
    """

    try:

        validate_image(input_path)

        logging.info(f"开始预处理图片：{input_path}")

        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("图片读取失败！")

        img_resized = cv2.resize(
            img,
            (512, 512),
            interpolation=cv2.INTER_LINEAR
        )

        cv2.imwrite(output_path, img_resized)

        logging.info(f"预处理完成，输出：{output_path}")

        return img_resized

    except Exception as e:

        logging.exception("图像预处理失败")

        raise


# ==============================
# LSB 信息嵌入
# ==============================

def lsb_encode(img_path, msg, output_path):
    """
    LSB信息隐藏
    """

    try:

        validate_image(img_path)

        validate_message(msg)

        check_capacity(img_path, msg)

        logging.info("开始LSB信息嵌入")

        img = Image.open(img_path).convert("RGB")

        width, height = img.size

        hide_msg = get_binary_msg(msg)

        length = len(hide_msg)

        count = 0

        img_data = np.array(img)

        for i in range(width):

            for j in range(height):

                if count >= length:
                    break

                pixel = list(img_data[j, i])

                for c in range(3):

                    if count < length:

                        pixel[c] = (
                            pixel[c] & ~1
                        ) | int(hide_msg[count])

                        count += 1

                img_data[j, i] = tuple(pixel)

            if count >= length:
                break

        stego_img = Image.fromarray(img_data)

        stego_img.save(output_path)

        logging.info(
            f"LSB嵌入完成，共写入{count} bit"
        )

        return output_path

    except Exception as e:

        logging.exception("LSB嵌入失败")

        raise


# ==============================
# LSB 信息提取
# ==============================

def lsb_decode(stego_path, msg_len):
    """
    提取LSB隐藏信息
    """

    try:

        validate_image(stego_path)

        if not isinstance(msg_len, int):

            raise TypeError("消息长度必须为整数")

        if msg_len <= 0:

            raise ValueError("消息长度必须大于0")

        logging.info("开始提取秘密信息")

        img = Image.open(stego_path).convert("RGB")

        width, height = img.size

        bit_len = msg_len * 8

        count = 0

        result_bin = ""

        img_data = np.array(img)

        for i in range(width):

            for j in range(height):

                if count >= bit_len:
                    break

                pixel = img_data[j, i]

                for c in range(3):

                    if count < bit_len:

                        result_bin += str(pixel[c] & 1)

                        count += 1

            if count >= bit_len:
                break

        byte_data = bytearray()

        for i in range(0, len(result_bin), 8):

            byte = result_bin[i:i + 8]

            byte_data.append(int(byte, 2))

        decoded_msg = byte_data.decode(
            "utf-8",
            errors="ignore"
        )

        logging.info("秘密信息提取成功")

        return decoded_msg

    except Exception as e:

        logging.exception("LSB提取失败")

        raise

# ==============================
# 位平面分析
# ==============================

def bit_plane_analysis(img_path):
    """
    位平面分析
    """

    try:

        validate_image(img_path)

        logging.info("开始位平面分析")

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("图片读取失败！")

        for i in range(8):

            mask = np.uint8(1 << i)

            bit_plane = (
                np.bitwise_and(img, mask) > 0
            ) * 255

            plt.figure(figsize=(6, 3))

            plt.subplot(1, 2, 1)
            plt.imshow(
                np.bitwise_and(
                    img,
                    np.bitwise_not(mask)
                ),
                cmap='gray'
            )
            plt.title(f'去掉第{i+1}位平面')

            plt.subplot(1, 2, 2)
            plt.imshow(bit_plane, cmap='gray')
            plt.title(f'仅第{i+1}位平面')

            plt.tight_layout()

            plt.show()

        logging.info("位平面分析完成")

    except Exception:

        logging.exception("位平面分析失败")

        raise


# ==============================
# DCT频域分析
# ==============================

def dct_distortion_demo(img_path):
    """
    DCT频域失真模拟
    """

    try:

        validate_image(img_path)

        logging.info("开始DCT分析")

        img = cv2.imread(
            img_path,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            raise ValueError("图片读取失败！")

        img_f32 = np.float32(img)

        img_dct = cv2.dct(img_f32)

        img_dct_copy = img_dct.copy()

        # 使用统一配置，避免Magic Number
        img_dct_copy[
            LOW_FREQUENCY_SIZE:,
            :
        ] = 0

        img_dct_copy[
            :,
            LOW_FREQUENCY_SIZE:
        ] = 0

        img_r = cv2.idct(img_dct_copy)

        img_r = np.clip(
            img_r,
            0,
            255
        ).astype(np.uint8)

        plt.figure(figsize=(10, 4))

        plt.subplot(131)
        plt.imshow(img, cmap='gray')
        plt.title("原始灰度图")

        plt.subplot(132)
        plt.imshow(
            np.log(np.abs(img_dct) + 1),
            cmap='hot'
        )
        plt.title("DCT系数(对数域)")

        plt.subplot(133)
        plt.imshow(img_r, cmap='gray')
        plt.title("DCT重构(失真后)")

        plt.tight_layout()

        plt.show()

        logging.info("DCT分析完成")

    except Exception:

        logging.exception("DCT分析失败")

        raise


# ==============================
# 主程序
# ==============================

if __name__ == "__main__":

    orig_name = "bupt.bmp"

    gray_name = "buptgray.bmp"

    stego_name = "buptgraystego.bmp"

    secret_text = "BUPTshahexiaoqu"

    try:

        logging.info("========== 程序开始 ==========")

        # 文件检查
        validate_image(orig_name)

        # 图像预处理
        preprocess_image(
            orig_name,
            gray_name
        )

        # 信息嵌入
        lsb_encode(
            gray_name,
            secret_text,
            stego_name
        )

        # 信息提取
        extracted = lsb_decode(
            stego_name,
            len(secret_text)
        )

        print("=" * 50)
        print("提取出的秘密信息：")
        print(extracted)
        print("=" * 50)

        logging.info(
            f"提取结果：{extracted}"
        )

        # PSNR计算
        img_orig = cv2.imread(gray_name)

        img_stego = cv2.imread(stego_name)

        val_psnr = psnr(
            img_orig,
            img_stego
        )

        print(
            f"嵌入后的PSNR：{val_psnr:.2f} dB"
        )

        logging.info(
            f"PSNR={val_psnr:.2f}"
        )

        # 位平面分析
        bit_plane_analysis(gray_name)

        # DCT分析
        dct_distortion_demo(gray_name)

        logging.info("程序运行结束")

    except Exception as e:

        print("\n程序运行失败！")
        print("错误信息：", e)

        logging.exception("程序运行异常")
