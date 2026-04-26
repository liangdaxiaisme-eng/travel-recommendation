#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载飞猪数据集
"""
import requests
import os
import time

# 阿里云OSS下载链接
urls = [
    "https://tianchi-jupyter-sh.oss-cn-shanghai.aliyuncs.com/file/opensearch/documents/113649/user_profile.csv?Expires=1775939280&OSSAccessKeyId=STS.NY8qGTKG3GuVupHvgjktdhHYu&Signature=hq%2Fudmsh5H%2BTfGYlxsrbj%2FWdP9g%3D&response-content-disposition=attachment%3B%20&security-token=CAIS0wJ1q6Ft5B2yfSjIr5qNOv3gppgS8Le9d1b5kmc%2FZ%2FtIh43ylzz2IHhMeXdvCeAfsvs1lG9W7vYYlrp6SJtIXleCZtF94oxN9h2gb4fb4zhsQDi708%2FLI3OaLjKm9u2wCryLYbGwU%2FOpbE%2B%2B5U0X6LDmdDKkckW4OJmS8%2FBOZcgWWQ%2FKBlgvRq0hRG1YpdQdKGHaONu0LxfumRCwNkdzvRdmgm4NgsbWgO%2Fks0aH1Q2rlbdM%2F9WvfMX0MvMBZskvD42Hu8VtbbfE3SJq7BxHybx7lqQs%2B02c5onAXQYBs0zabLCErYM3fFVjGKE5H6Nft73wheU9sevOjZ%2F6jg5XOu1FguwGsiZLaaEuccPe1bZRHd6TUxylWSCUrAggMK0O1dMK5q47msugW56BMxZYrhuwxOFJFi407GlNJVaL3m9IH%2FgR7RHcmfDuQfznsPNHHXpwUvcagAFxJgV4TA7ar8zebpCfJnwCCIZI1%2BXngS6poVvy5NZzIZx8PW2QgUvx5QOIMgbZyscmtlCKnt6lqwZOA6mt9AoR7W1tu%2Fytv3YN8oc8wqFzGXbUkD66lY0RwRvhIg1ZE51dJKBTwNmMIh0tE1N4b70Di1mMJbmfLe8yL6RTy3T3uCAA",
    "https://tianchi-jupyter-sh.oss-cn-shanghai.aliyuncs.com/file/opensearch/documents/113649/item_profile.csv?Expires=1775939295&OSSAccessKeyId=STS.NY8qGTKG3GuVupHvgjktdhHYu&Signature=l6q43ZnpyI%2FqKZdX27uTIuYcwR0%3D&response-content-disposition=attachment%3B%20&security-token=CAIS0wJ1q6Ft5B2yfSjIr5qNOv3gppgS8Le9d1b5kmc%2FZ%2FtIh43ylzz2IHhMeXdvCeAfsvs1lG9W7vYYlrp6SJtIXleCZtF94oxN9h2gb4fb4zhsQDi708%2FLI3OaLjKm9u2wCryLYbGwU%2FOpbE%2B%2B5U0X6LDmdDKkckW4OJmS8%2FBOZcgWWQ%2FKBlgvRq0hRG1YpdQdKGHaONu0LxfumRCwNkdzvRdmgm4NgsbWgO%2Fks0aH1Q2rlbdM%2F9WvfMX0MvMBZskvD42Hu8VtbbfE3SJq7BxHybx7lqQs%2B02c5onAXQYBs0zabLCErYM3fFVjGKE5H6Nft73wheU9sevOjZ%2F6jg5XOu1FguwGsiZLaaEuccPe1bZRHd6TUxylWSCUrAggMK0O1dMK5q47msugW56BMxZYrhuwxOFJFi407GlNJVaL3m9IH%2FgR7RHcmfDuQfznsPNHHXpwUvcagAFxJgV4TA7ar8zebpCfJnwCCIZI1%2BXngS6poVvy5NZzIZx8PW2QgUvx5QOIMgbZyscmtlCKnt6lqwZOA6mt9AoR7W1tu%2Fytv3YN8oc8wqFzGXbUkD66lY0RwRvhIg1ZE51dJKBTwNmMIh0tE1N4b70Di1mMJbmfLe8yL6RTy3T3uCAA",
    "https://tianchi-jupyter-sh.oss-cn-shanghai.aliyuncs.com/file/opensearch/documents/113649/user_item_behavior_history.csv?Expires=1775939302&OSSAccessKeyId=STS.NZDMKiT2y9Z9Lf8yygMhHUqfd&Signature=krkuN4fJvdwyGEy%2BP9SdldojiQU%3D&response-content-disposition=attachment%3B%20&security-token=CAIS0wJ1q6Ft5B2yfSjIr5nxBvHdue1YjpjSTkCJnXkyQedkurTNhjz2IHhMeXdvCeAfsvs1lG9W7vYYlrp6SJtIXleCZtF94oxN9h2gb4fb428rBDi708%2FLI3OaLjKm9u2wCryLYbGwU%2FOpbE%2B%2B5U0X6LDmdDKkckW4OJmS8%2FBOZcgWWQ%2FKBlgvRq0hRG1YpdQdKGHaONu0LxfumRCwNkdzvRdmgm4NgsbWgO%2Fks0aH1Q2rlbdM%2F9WvfMX0MvMBZskvD42Hu8VtbbfE3SJq7BxHybx7lqQs%2B02c5onAXQYBs0zabLCErYM3fFVjGKE5H6Nft73wheU9sevOjZ%2F6jg5XOu1FguwGsiZLaaEuccPe1bZRHd6TUxylWYA66RVDrght5%2F5PgYmu1w69ztNUSEVKZHizAdSM0zPpeWyIIIvei5pV6vcbgiuuk5PkSFbn%2F6F6k3pwUvcagAF6RJZdfSMvV7RpKAvp%2F2wO6mjSZ6xs1WmewYVYgnjWzVmBaQSenMpCHml4GAOt79uiPcOnV14%2FAAlTOA%2BNezgOabfAvJ0gkYH55I1Dw%2Bfs%2FKGsrYRiBTDWQCqVq7LazGxTh0N%2FeioMWewGYMnB%2B%2FNUB3GETVfeZXt53rbD7oWECCAA"
]

download_dir = "/home/asd/论文资料/4/旅游推荐数据集/训练代码/data"

os.makedirs(download_dir, exist_ok=True)

for i, url in enumerate(urls):
    filename = os.path.join(download_dir, [f"user_profile.csv", f"item_profile.csv", f"user_item_behavior_history.csv"][i])
    print(f"\n[{i+1}/3] 下载 {filename}...")
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r  进度: {progress:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)

        print(f"\n  ✓ 下载完成: {os.path.getsize(filename)} bytes")
        time.sleep(2)  # 避免请求过快

    except Exception as e:
        print(f"\n  ✗ 下载失败: {e}")
        continue

print("\n\n所有文件下载完成!")
