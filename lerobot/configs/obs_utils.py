import os
import logging
import shutil
import tarfile
import traceback

from urllib.parse import urlparse
from obs import ObsClient


def get_env_variable(name, default=None):
    """读取环境变量，如果未设置则返回默认值"""
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"The env variable '{name}' is not set and has no default value!")
    return value


class EnvConfig:
    def __init__(self):
        # obs
        self.ACCESS_KEY_ID = get_env_variable("ACCESS_KEY_ID")
        self.SECRET_ACCESS_KEY = get_env_variable("SECRET_ACCESS_KEY")
        self.OBS_SERVER = get_env_variable("OBS_SERVER")
        self.POLICY_CKPT_DIR = get_env_variable("POLICY_CKPT_DIR", "checkpoints")


def download_policy_file(obs_url: str) -> str:
    env_config = EnvConfig()
    obsClient = ObsClient(access_key_id=env_config.ACCESS_KEY_ID,
                          secret_access_key=env_config.SECRET_ACCESS_KEY,
                          server=env_config.OBS_SERVER)
    dest_dir = env_config.POLICY_CKPT_DIR

    # --- 1. Parse obs_url ---
    if not obs_url.lower().startswith("obs://"):
        raise ValueError("Invalid obs_url. Must start with 'obs://'.")

    path = obs_url[6:]
    parts = path.split('/', 1)
    if len(parts) != 2:
        raise ValueError("obs_url must be in format: obs://bucket/prefix")

    bucket_name = parts[0]
    prefix = parts[1]
    logging.info(f"bucket_name: {bucket_name}, prefix: {prefix}")
    if not prefix.endswith('/'):
        prefix += '/'

    # --- 2. Compute final root path ---
    root_dir_name = os.path.basename(prefix.rstrip('/'))
    final_dest_dir = os.path.join(dest_dir, root_dir_name)
    os.makedirs(final_dest_dir, exist_ok=True)

    # --- 3. List objects ---
    response = obsClient.listObjects(bucket_name, prefix)
    objects = response.get("body", {}).get("contents", [])

    # --- 4. Download all files ---
    for obj in objects:
        key = obj.get("key", "")
        size = obj.get("size", 0)

        # Relative path under the prefix
        relative_path = key[len(prefix):]
        local_path = os.path.join(final_dest_dir, relative_path)

        if key.endswith("/") and size == 0:
            os.makedirs(local_path, exist_ok=True)
            logging.info(f"[Folder] Created local directory: {local_path}")
        else:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            file_name = os.path.basename(local_path)
            local_dir = os.path.dirname(local_path)
            logging.info(f"[File] Downloading {key} -> {local_path}")
            try:
                download_obs_file(
                    obsClient=obsClient,
                    bucket_name=bucket_name,
                    object_key=key,
                    file_name=file_name,
                    dest_dir=local_dir
                )
            except Exception as e:
                logging.info(f'Download File Failed {e}')
                logging.info(traceback.format_exc())

    return final_dest_dir


def download_obs_file(obsClient: ObsClient, bucket_name: str, object_key: str, file_name: str, dest_dir: str,
                      enable_checkpoint: bool = False):
    download_file = os.path.join(dest_dir, file_name)
    remove_dir_or_file(download_file)
    # 分段下载的并发数
    task_num = 5
    # 分段的大小
    part_size = 10 * 1024 * 1024
    # 断点续传下载对象
    resp = obsClient.downloadFile(bucket_name, object_key, download_file,
                                       part_size, task_num, enable_checkpoint)
    # 返回码为2xx时，接口调用成功，否则接口调用失败
    if resp.status < 300:
        logging.info('Download File Succeeded')
        logging.info(f'requestId: {resp.requestId}')
    else:
        logging.info('Download File Failed')
        logging.info(f'requestId: {resp.requestId}')
        logging.info(f'errorCode: {resp.errorCode}', )
        logging.info(f'errorMessage: {resp.errorMessage}')

    return download_file


def remove_dir_or_file(target_path):
    """删除路径对应的文件"""
    if not os.path.exists(target_path):
        return
    logging.warning(f"{target_path} will be remove.")
    if os.path.isdir(target_path):
        shutil.rmtree(target_path)
    else:
        os.remove(target_path)


def parse_obs_url(obs_url: str):
    """解析华为云 OBS URL，提取 bucketName、objectKey 和 fileName"""
    parsed_url = urlparse(obs_url)

    # 解析 bucketName（通常是二级域名部分）
    domain_parts = parsed_url.netloc.split('.')
    if len(domain_parts) > 1:
        bucket_name = domain_parts[0]
    else:
        raise ValueError("无法解析 bucketName")

    # 解析 objectKey（去掉前导 '/'）
    object_key = parsed_url.path.lstrip('/')

    # 解析 fileName（objectKey 的最后一部分）
    file_name = object_key.split('/')[-1] if object_key else ''

    return bucket_name, object_key, file_name


def is_tar_gz_file(file_path: str):
    """判断文件是否是 .tar.gz 结尾的压缩文件"""
    return file_path.endswith(".tar.gz")


def extract_tar_gz(file_path: str, dest_dir: str = None):
    """解压 .tar.gz 文件到指定目录，如果 dest_dir 为 None，则解压到当前目录"""
    if not is_tar_gz_file(file_path):
        raise ValueError("文件不是 .tar.gz 格式")

    if dest_dir is None:
        dest_dir = os.getcwd()  # 设为当前目录

    os.makedirs(dest_dir, exist_ok=True)  # 确保目标目录存在

    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)

    logging.info(f"文件 {file_path} 已解压到 {dest_dir}")


def contains_obs_url(s: str) -> bool:
    if not isinstance(s, str):
        return False  # 非字符串直接返回 False
    return 'obs://' in s.strip().lower()
