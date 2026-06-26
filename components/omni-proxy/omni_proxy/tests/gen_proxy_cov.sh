#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOP_DIR="$(cd "$WORK_DIR/../../" && pwd)"
echo "TOP_DIR is ${TOP_DIR}"
REPORT_PATH="proxy_report"

mkdir -p ${REPORT_PATH}
rm -rf ./${REPORT_PATH}/*
gcovr --gcov-ignore-errors=no_working_dir_found --root ${TOP_DIR}/nginx-1.28.0 --filter "${TOP_DIR}/omni_proxy/.*\.c$" --html --html-details --html=./${REPORT_PATH}/coverage_report.html  --xml --xml=./${REPORT_PATH}/coverage_report.xml --txt-summary
tar czf proxy_cov.tar.gz ./${REPORT_PATH}
