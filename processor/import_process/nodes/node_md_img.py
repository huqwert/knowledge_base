# processor/import_process/nodes/node_md_img.py
import base64
import json
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Tuple, List, Deque, Dict

from langchain_openai import ChatOpenAI
from minio.deleteobjects import DeleteObject

from processor.config.config import lm_config, minio_config
from processor.import_process.base import NodeBase
from processor.import_process.prompt import IMAGE_SUMMARY
from processor.import_process.state import ImportGraphState
from processor.tool.logger import logger
from processor.utils.minio_utils import get_minio_client


class NodeMDImg(NodeBase):
    """
    MarkDown图片处理节点：多模态图片理解
    """

    name = "node_md_img"

    def process(self, state: ImportGraphState):

        # 步骤1:获取md内容,文件path,图片path
        md_content, md_path_obj, images_dir = self._step1_get_content(state)
        if not images_dir.exists():
            # 直接跳过, 无需处理
            logger.info(f"【图片处理】图片目录不存在：{images_dir}")
            return {
                "md_content": md_content
            }

        # 步骤2:扫描并筛选md文档中引用的所有的图片
        images = self._step2_scan_images(md_content, images_dir)
        if not images:
            # 直接跳过, 无需处理
            logger.info(f"【图片处理】无图片需要处理")
            return {
                "md_content": md_content
            }

        # 步骤3:生成图片摘要(VLM)
        file_title = md_path_obj.stem
        summaries = self._step3_generate_summaries(file_title, images)

        # 步骤4:上传图片至MinIO,填充图片摘要和路径
        new_md_content = self._step4_upload_and_replace(file_title, images, summaries, md_content)

        # 步骤4.5: 将新的内容存入物理文件
        new_md_file_name = self._step5_backup_new_md_file(state['md_path'], new_md_content)

        # 步骤5:返回结果
        return {
            "md_content": new_md_content,
            "md_path": new_md_file_name
        }

    def _step1_get_content(self, state: ImportGraphState) -> tuple[str, Path, Path]:
        # 1. 参数的非空校验
        md_path = state.get("md_path")
        if not md_path:
            raise ValueError("md_path is required")

        # 2. 将路径类型转换成Path
        md_path_obj = Path(md_path)

        # 3. 判断目标的md文件是否存在
        if not md_path_obj.exists():
            raise FileNotFoundError(f"指定的MD文件不存在：{md_path}")

        # 4. 获取md_content
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # 5.组装图片路径
        images_dir = md_path_obj.parent / "images"

        return md_content, md_path_obj, images_dir

    def _step2_scan_images(self, md_content: str, images_dir: Path) -> List[Tuple[str, str, Tuple[str, str]]]:

        # 1. MinIO支持的图片格式集合（小写后缀，统一匹配标准）
        IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

        # 2. 定义待处理图片列表
        images = []

        # 3. 遍历图片目录
        for image_file in os.listdir(images_dir):
            logger.info(f"【图片处理】处理图片：{image_file}")

            # 3.1 过滤无效后缀
            file_ext = Path(image_file).suffix.lower()
            if file_ext not in IMAGE_EXTENSIONS:
                logger.warning(f"【图片处理】无效的图片后缀：{image_file}")
                continue

            # 3.2 查找图片的前文和后文(元组)
            context = self._find_image_in_md(md_content, image_file)

            image_path = str(images_dir / image_file)

            # image_file 图片文件名
            # image_path 图片路径
            # image_content 图片上下文
            # 3.3 判断图片是否被md文档引用
            if not context:
                logger.warning(f"【图片处理】图片未被引用：{image_file}")
                continue
            images.append((image_file, image_path, context))



        return images

    def _find_image_in_md(self, md_content: str, image_file: str,content_len: int = 100):
        """
        查找MD内容中指定图片的所有引用位置，并返回每个位置的上下文文本
        """
        # 1. 定义正则表达式
        # ![描述](xxx文件名.扩展名)
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")

        # 2. 匹配图片:找到一个既返回
        match = pattern.search(md_content)
        if not match:
            logger.warning(f"【图片处理】未找到图片：{image_file}")
            return None

        start, end = match.span()
        # 3. 获取图片的前文和后文
        pre_text = md_content[max(0,start - content_len):start] # 前文
        post_text = md_content[end:min(end + content_len, len(md_content))] #后文
        logger.info(f"{image_file}的前文: {pre_text}")
        logger.info(f"{image_file}的后文: {post_text}")

        return pre_text, post_text

    def _step3_generate_summaries(self, file_title: str, images: List[Tuple[str, str, Tuple[str, str]]]) -> dict:
        """
        为图片生成内容摘要,调用VLM模型,设置速率限制
        :param file_title:
        :param images:
        :return: 摘要字典 {键:图片文件名, 值:摘要}
        """

        summarise = {}
        request_deque = deque()
        for image_file, image_path, context in images:
            # 2.1 速率限制
            self._apply_api_rate_limit(request_deque, max_requests=50)

            # 2.2 调用模型生成摘要
            summarise[image_file] = self._summarize_image(image_path=image_path, file_title=file_title, context=context)

            # for i in range(100):
            #     # 2.1 速率限制
            #     self._apply_api_rate_limit(request_deque, max_requests=10)
            #     logger.info(f"【模型调用】：{i}")
            #
            #     # 2.2 调用模型生成摘要
            #     time.sleep(2)

        return summarise

    def _apply_api_rate_limit(self, request_deque: Deque[float], max_requests: int, window_size: int = 60):
        """"
        通用滑动窗口API限速器
        基本思想:
        维护一个存储请求时间戳信息的双端队列,可以入队和出队
        当单位时间窗口(window_size)内请求时间戳的数量超过上限(max_requests)时,则停止入队,等待直到可以入队
        """
        # 1. 获取当前时间戳
        current_time = time.time()

        # 2. 移除过期的请求时间戳
        while request_deque and current_time - request_deque[0] >= window_size:
            request_deque.popleft()

        # 3. 如果窗口内请求数量达上限,计算阻塞等待的剩余时间
        if len(request_deque) >= max_requests:

            # 计算剩余等待时间
            sleep_duration = window_size - (current_time - request_deque[0])
            if sleep_duration > 0:
                logger.info(f"【API调用】API请求限速,等待{sleep_duration:.2f}秒")
                time.sleep(sleep_duration)

                # 等待后,更新当前时间
                current_time = time.time()
                while request_deque and current_time - request_deque[0] >= window_size:
                    request_deque.popleft()
        # 4. 入队
        request_deque.append(current_time)
        logger.info(f"【API调用】API请求时间已记录,当前窗口内的请求数量: {len(request_deque)}")

    def _summarize_image(self, image_path: str, file_title: str, context: Tuple[str, str]):
        """
        调用VLM模型对图片内容进行总结
        :param image_path:
        :param file_title:
        :param context:
        :return:
        """
        # 1. 将图片转换成base64编码
        with open(image_path,"rb") as f:
            image_data = f.read()
        prompt = IMAGE_SUMMARY.format(
            file_title=file_title,
            context=context
        )
        try:

            image_base64 = base64.b64encode(image_data).decode('utf-8')
            # 2. 调用模型
            chat_model = ChatOpenAI(
                model=lm_config.vl_model,
                api_key=lm_config.api_key,
                base_url=lm_config.base_url,
                temperature=lm_config.lm_temperature
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
            response = chat_model.invoke(messages)
            return response.content.strip().replace("\n", "")
        except:
            logger.error(f"【模型调用】图片摘要总结失败：{image_path}")
            return "图片描述"

    def _step4_upload_and_replace(
            self,
            file_title: str,
            images: List[Tuple[str, str, Tuple[str, str]]],
            summaries: Dict[str, str],
            md_content: str
    )-> str:
        """
        Step4: 上传图片并合并信息，然后替换 Markdown 中的内容。

        流程：
        1. 确定 MinIO 上的上传目录（按文档名隔离）。
        2. 清理该目录下的旧数据。
        3. 批量上传图片。
        4. 合并“图片摘要”和“图片URL”。
        5. 替换 Markdown 文本中的图片引用。
        """
        # 1. 构造上传目录，去除文件名中的空格
        minio_img_dir = minio_config.img_dir
        upload_dir = f"{minio_img_dir}/{file_title}".replace(" ", "")

        # 2. 清理旧数据
        self._clean_minio_directory(upload_dir)

        # 3. 上传新图片，获取URL映射
        urls = self._upload_images_batch(upload_dir, images)

        # 4. 合并图片摘要和URL
        image_info = self._merge_summary_and_url(summaries, urls)

        # 5. 替换MD内容中的本地图片引用为MinIO远程引用
        md_content =self._process_md_file(md_content, image_info)
        return md_content



    def _clean_minio_directory(self, upload_dir: str) -> None:
        """
        注意：删除业务是非核心业务，一旦失败不要影响主业务执行

        幂等性清理：上传前先删除 MinIO 中指定目录下的旧文件。
        防止重名文件导致的内容混淆或垃圾堆积。
        :param upload_dir: 要清理的目录路径
        """
        try:
            # 1. 获取MinIO客户端对象
            minio_client = get_minio_client()

            # 2. 查找要删除的对象，并组装对象列表
            # 写法一
            # objects_to_delete = minio_client.list_objects(minio_config.bucket_name,upload_dir,recursive=True)
            # delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]

            # 写法二
            delete_list = map(lambda obj: DeleteObject(obj.object_name),
                              minio_client.list_objects(minio_config.bucket_name,
                            upload_dir,recursive=True))
            # 3. 批量删除
            if delete_list:
                errors = minio_client.remove_objects(
                    minio_config.bucket_name,
                    delete_list
                )
                for error in errors:
                    logger.error(f"【MinIO删除】对象删除失败: {error}")

        except Exception as e:
            logger.error(f"MinIO清理目录失败: {e}")

    def _upload_images_batch(
            self,
            upload_dir: str,
            images: List[Tuple[str, str, Tuple[str, str]]]
    ) -> Dict[str, str]:
        """
        批量上传待处理图片至MinIO，返回图片文件名与访问URL的映射关系
        :param upload_dir: MinIO上传根目录
        :param images: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
        :return: 图片URL字典，键：图片文件名，值：MinIO访问URL
        """
        urls = {}
        for image_file, image_path, context in images:

            # 将文件上传并获取这个图片的url地址
            # 注意 minio中的 object_name 是除了 协议://主机:端口/bucketName/  后面的路径地址
            object_name = f"{upload_dir}/{image_file}"
            urls[image_file] = self._upload_to_minio(image_path, object_name)


        return urls

    def _upload_to_minio(self, image_path: str, object_name:str) -> str:

        """
        图片上传到MinIO
        :param image_path: 图片的本地绝对路径
        :param object_name: 图片在MinIO上的 object name
        :return: url地址
        """

        # 1. 获取MinIO Client对象
        minio_client = get_minio_client()

        # 2. 上传图片
        minio_client.fput_object(
            minio_config.bucket_name,
            object_name,
            image_path
        )

        # 3. 获取图片的访问URL
        url = f"http://{minio_config.endpoint}/{minio_config.bucket_name}/{object_name}"
        return url

    def _merge_summary_and_url(
            self,
            summaries: Dict[str, str],
            urls: Dict[str, str]
    ) -> Dict[str, Tuple[str, str]]:


        image_info = {}
        for images_file, summary in summaries.items():
            image_info[images_file] = (summary, urls[images_file])

        return image_info

    def _process_md_file(self, md_content: str, image_info: Dict[str, Tuple[str, str]]) -> str:
        """
        核心功能：替换MD内容中的本地图片引用为MinIO远程引用
        替换规则：![原描述](本地路径) → ![图片摘要](MinIO访问URL)
        :param md_content: 原始MD文件内容
        :param image_info: 合并后的图片信息字典，键：图片文件名，值：(摘要, URL)
        :return: 替换后的新MD内容
        """
        for image_file, (summary, url) in image_info.items():
            # 1、定义正则表达式
            # ![描述](xxx文件名.扩展名)
            pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r"\)")

            # 2、替换
            # lambda: 防御性编程
            md_content = pattern.sub(lambda x: f"![{summary}]({url})", md_content)
        logger.info(f"summary和url替换完成")
        return md_content

    def _step5_backup_new_md_file(self, origin_md_path: str, new_md_content: str) -> str:
        """
       步骤5：将处理后的MD内容保存为新文件（原文件不变，避免数据丢失）
       新文件命名规则：原文件名 + _new.md（如test.md → test_new.md）
       :param origin_md_path: 原始MD文件完整路径
       :param new_md_content: 处理后的新MD内容
       :return: 新MD文件的完整路径
       """
        # 构造新文件路径：替换原后缀为 _new.md
        new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"

        # 写入新MD内容（覆盖写入，若文件已存在则更新）
        with open(new_md_file_name, "w", encoding="utf-8") as f:
            f.write(new_md_content)

        logger.info(f"处理后md文件已保存，新文件路径:{new_md_file_name}")

        return new_md_file_name


