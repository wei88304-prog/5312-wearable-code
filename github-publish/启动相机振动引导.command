#!/bin/bash
cd "$(dirname "$0")" || exit 1
./.venv/bin/python sony_wifi_yolo.py --testing --haptics-port auto --haptics-case 2 --pulse-gap 0.3 "$@"
task_exit_code=$?
if [ "$task_exit_code" -ne 0 ]; then
  read -r -p "按回车关闭窗口。"
fi
exit "$task_exit_code"
