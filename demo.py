"""
最小调用示例：把一张拍书的照片展平为矩形书页。

用法：
    python demo.py 输入.jpg 输出.jpg

若省略参数，则尝试读取同目录下的 input.jpg 并写出 output.jpg。
"""
import sys

from book_flatten import correct_paper, PageNotFoundError


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "input.jpg"
    dst = sys.argv[2] if len(sys.argv) > 2 else "output.jpg"

    with open(src, "rb") as file:
        image_bytes = file.read()

    try:
        corrected_bytes = correct_paper(image_bytes)
    except PageNotFoundError:
        print(f"未检测到书页：{src}")
        return

    with open(dst, "wb") as file:
        file.write(corrected_bytes)

    print(f"已保存：{dst}（{len(corrected_bytes)} 字节）")


if __name__ == "__main__":
    main()
