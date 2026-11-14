#!/bin/bash
cd "$(dirname "$0")" || exit 1
./.venv/bin/python sony_wifi_yolo.py --choose-case --haptics-port auto "$@"
echo "测试结束，按回车关闭窗口。"
read -r
