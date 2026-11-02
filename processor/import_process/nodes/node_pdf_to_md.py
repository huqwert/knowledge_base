# processor/import_process/nodes/node_pdf_to_md.py
import json
import os
import shutil
import time
from pathlib import Path
from zipfile import ZipFile

import requests
from dotenv import load_dotenv

from processor.import_process.base import NodeBase
from processor.import_process.state import ImportGraphState
from processor.tool.logger import logger


env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.env"))
load_dotenv(dotenv_path=env_path, override=True)


class NodePDFToMD(NodeBase):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """

    name = "node_pdf_to_md"

    def process(self, state: ImportGraphState):

        # 步骤1：校验输入参数
        pdf_path_obj, output_dir_obj = self._step1_validate_paths(state)

        # 步骤2：上传pdf到MinerU并轮询解析结果
        zip_url = self._step2_upload_and_poll(pdf_path_obj)

        # 步骤3：下载zip包并解压
        file_title = pdf_path_obj.stem
        md_path = self._step3_download_and_extract(zip_url, output_dir_obj, file_title)

        # # 步骤4：读取md的内容
        # with open(md_path, "r", encoding="utf-8") as f:
        #     md_content = f.read()

        # 步骤5：返回结果
        return {
            "md_path": md_path,
            # "md_content": md_content
        }

    def _step1_validate_paths(self, state):

        # 1. 参数的非空校验
        pdf_path = state.get("pdf_path")
        local_dir = state.get("local_dir")

        if not pdf_path:
            raise ValueError("请指定PDF文件路径")
        if not local_dir:
            raise ValueError("请指定输出目录")

        # 2. 将路径类型转换成Path
        pdf_path_obj = Path(pdf_path)
        output_dir_obj = Path(local_dir)

        # 3. 判断目标的pdf文件是否存在
        if not pdf_path_obj.exists():
            raise ValueError(f"PDF文件不存在：{pdf_path_obj}")

        # 4. 判断输出目录是否存在，如果不存在则创建
        if not output_dir_obj.exists():
            # parents=True：递归创建所有不存在的目录和子目录
            # exist_ok=True：如果目录已经存在，则不报错
            output_dir_obj.mkdir(parents=True, exist_ok=True)

        # 5. 返回结果
        return pdf_path_obj,output_dir_obj



    # 使用绝对路径加载 .env

    @staticmethod
    def _get_api_token():


        token = os.getenv("MINERU_API_TOKEN")
        if not token:
            raise ValueError("请设置MINERU_API_TOKEN环境变量")
        return token

    @staticmethod
    def _get_base_url():

        base_url = os.getenv("MINERU_BASE_URL")
        if not base_url:
            raise ValueError("请设置MINERU_BASE_URL环境变量")
        return base_url

    def _step2_upload_and_poll(self, pdf_path_obj):
        """
        step2：上传pdf到MinerU并轮询解析结果
        """
        # 1. 申请文件上传链接
        # 1.1 准备上传参数
        token = self._get_api_token()
        url = f"{self._get_base_url()}/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm"
        }
        # 1.2 发送请求
        response = requests.post(url, headers=header, json=data)

        # 1.3. 响应结果校验
        if response.status_code != 200:
            raise ValueError(f"申请文件上传链接失败：{response.text}")

        # 1.4 获取响应结果
        result = response.json()
        logger.info(f"响应成功. result:{result}")

        if result["code"] != 0:
            raise RuntimeError(f"申请文件上传链接失败：{result['msg']}")

        # 1.5 获取任务id
        batch_id = result["data"]["batch_id"]

        # 1.6 获取上传链接
        signed_url = result["data"]["file_urls"][0]
        logger.info(f"获取上传链接成功. batch_id:{batch_id}, signed_url:{signed_url}")

        # 2. 执行文件上传
        with open(pdf_path_obj, "rb") as f:
            res_upload = requests.put(signed_url, data=f)
            # 2.1 检查上传结果
            if res_upload.status_code != 200:
                raise RuntimeError(f"上传文件失败：{res_upload}，状态码：{res_upload.status_code}")

            # 2.2 上传成功
            logger.info("文件上传成功")

        # 3. 批量获取任务结果
        # 3.1 组装轮询url
        url = f"{self._get_base_url()}/extract-results/batch/{batch_id}"

        # 3.2 设置轮询开始时间
        start_time = time.time()
        timeout_seconds = 300 # 最大超时时间
        poll_interval = 3 # 轮询间隔
        logger.info(f"开始轮询任务结果. 最大超时时间:{timeout_seconds}s, 轮询间隔:{poll_interval}")

        # 3.3 开始轮询
        while True:

            # 3.3.1 记录已经消耗的时间
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise RuntimeError(f"【轮询中】任务超时，任务处理超过{timeout_seconds}s！batch_id:{batch_id}")

            # 3.3.2 提交请求
            try:
                res = requests.get(url, headers=header,timeout=5)
            except Exception as e:
                logger.warning(f"【轮询中】网络请求异常:{str(e)}，{poll_interval}s后重试。batch_id:{batch_id}")
                time.sleep(poll_interval)
                continue

            # 3.3.3 响应结果校验
            if res.status_code != 200:
                raise RuntimeError(f"【轮询中】获取任务结果失败：{res}，状态码：{res.status_code}")

            # 3.3.4 获取响应结果
            poll_data = res.json()
            if poll_data["code"] != 0:
                raise RuntimeError(f"【轮询中】获取任务结果失败：{poll_data['msg']}")

            # 3.3.5 解析结果
            result_item = poll_data["data"]["extract_result"][0]
            data_state = result_item["state"]

            # 3.3.6 根据状态判断是否结束轮询
            if data_state == "done":
                logger.info(f"【轮询结束】任务完成，总耗时：{elapsed_time:.2f}s。batch_id:{batch_id}")

                full_zip_url = result_item["full_zip_url"]
                logger.info(f"【轮询结束】返回ZIP包下载链接：{full_zip_url}。batch_id:{batch_id}")

                return full_zip_url
            elif data_state == "failed":
                err_msg = result_item.get("err_msg","未知错误")
                raise RuntimeError(f"【轮询失败】任务失败，原因：{err_msg}")

            else:
                logger.info(f"【轮询中】任务进行中，继续轮询...。已耗时：{elapsed_time:.2f}s。请稍后。batch_id:{batch_id}")
                time.sleep(poll_interval)

    def _step3_download_and_extract(self, zip_url, output_dir_obj, file_title):
        # 1. 下载
        logger.info(f"开始下载ZIP包. zip_url:{zip_url}, 保存路径:{output_dir_obj}, 文件标题:{file_title}")
        response = requests.get(zip_url)

        if response.status_code != 200:
            raise RuntimeError(f"【下载失败】下载失败，状态码：{response.status_code},响应结果：{response}")

        # 2. 保存
        zip_save_path = output_dir_obj / f"{file_title}.zip"
        with open(zip_save_path, "wb") as f:
            f.write(response.content)
        logger.info(f"【下载成功】ZIP包保存路径：{zip_save_path}")

        # 3. 删除已存在的解压目录
        unzip_dir_obj = output_dir_obj / file_title
        if unzip_dir_obj.exists():
            logger.info(f"【解压】目标目录已存在，删除目录：{unzip_dir_obj}")
            shutil.rmtree(unzip_dir_obj)

        # 4. 创建解压目录
        unzip_dir_obj.mkdir(parents=True, exist_ok=True)
        logger.info(f"【解压】创建解压目录：{unzip_dir_obj}")

        # 5. 解压
        logger.info(f"【解压】开始解压：{file_title}.zip")
        with ZipFile(zip_save_path, "r") as zip_file:
            zip_file.extractall(unzip_dir_obj)
        logger.info(f"【解压】解压完成. 解压目录:{unzip_dir_obj}")

        # 6. 重命名文件：full.md -> 文件名.md
        md_file_obj = unzip_dir_obj / "full.md"
        new_md_path = md_file_obj.with_name(file_title + ".md")
        md_file_obj.rename(new_md_path)
        logger.info(f"【MD重命名】重命名成功：{md_file_obj} -> {new_md_path}")

        return str(new_md_path.resolve())






















