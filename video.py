import json
import base64
import time
import random
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.tts.v20190823 import tts_client, models


class TencentTTSClient:
    def __init__(self, secret_id, secret_key, region="ap-beijing"):
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.region = region
        self.client = self._init_client()

    def _init_client(self):
        cred = credential.Credential(self.secret_id, self.secret_key)
        httpProfile = HttpProfile()
        httpProfile.endpoint = "tts.tencentcloudapi.com"
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        client = tts_client.TtsClient(cred, self.region, clientProfile)
        return client

    def text_to_voice(self, text, session_id="session-1234", volume=5, speed=0, project_id=0, model_type=1, voice_type=301030, primary_language=1):
        try:
            req = models.TextToVoiceRequest()
            params = {
                "Text": text,
                "SessionId": session_id,
                "Volume": volume,
                "Speed": speed,
                "ProjectId": project_id,
                "ModelType": model_type,
                "VoiceType": voice_type,
                "PrimaryLanguage": primary_language
            }
            req.from_json_string(json.dumps(params))
            resp = self.client.TextToVoice(req)
            print(resp.to_json_string())
            audio = resp.Audio.encode()
            timestamp = int(time.time() * 1000)  # 获取当前时间戳，单位为毫秒
            random_part = random.randint(1000, 9999)  # 生成一个1000到9999之间的随机数
            file_path = f"./{timestamp}_{random_part}.mp3"
            with open(file_path, "wb") as f:
                f.write(base64.decodebytes(audio))
            return file_path
        except TencentCloudSDKException as err:
            print(err)
