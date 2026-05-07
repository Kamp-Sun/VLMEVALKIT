import os
from typing import Any, Dict, List, Optional, Tuple, Union

from vlmeval.api.base import BaseAPI
from vlmeval.smp import encode_image_to_base64


def _to_data_url_if_local(image: str) -> str:
    """Ark Responses API expects `image_url`. Support http(s), data-url, and local file -> data-url."""
    if image.startswith(("http://", "https://", "data:image/")):
        return image
    if os.path.exists(image):
        from PIL import Image

        img = Image.open(image)
        b64 = encode_image_to_base64(img)
        return f"data:image/jpeg;base64,{b64}"
    raise ValueError(f"Invalid image path/url: {image}")


class ArkResponsesAPI(BaseAPI):
    """
    Volcengine Ark Responses API wrapper via OpenAI SDK.

    Why needed:
      Some Doubao/Seed multimodal models are accessible via `/api/v3/responses`
      but not via `/api/v3/chat/completions` for a given account/key.
    """

    is_api: bool = True

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        retry: int = 3,
        verbose: bool = False,
        system_prompt: Optional[str] = None,
        timeout: int = 120,
        max_output_tokens: int = 32768,
        thinking: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # Lazy import to avoid hard dependency at import-time for users who don't use Ark.
        from openai import OpenAI

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.thinking = thinking

        if api_key is None:
            # Prefer official env var name; keep compatibility with this repo's existing variable.
            api_key = os.getenv("ARK_API_KEY") or os.getenv("DOUBAO_VL_KEY")
        assert api_key, "Please set ARK_API_KEY (preferred) or DOUBAO_VL_KEY for Ark access."
        self.api_key = api_key

        # NOTE: do NOT log api_key
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    def _build_input(self, inputs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for item in inputs:
            if item["type"] == "image":
                content.append({"type": "input_image", "image_url": _to_data_url_if_local(item["value"])})
            elif item["type"] == "text":
                content.append({"type": "input_text", "text": item["value"]})
            else:
                raise ValueError(f"Unsupported type for Ark Responses: {item}")
        if self.system_prompt:
            content.insert(0, {"type": "input_text", "text": self.system_prompt})
        return [{"role": "user", "content": content}]

    def generate_inner(self, inputs: Union[str, List[Dict[str, str]]], **kwargs) -> Tuple[int, str, str]:
        # BaseAPI will preproc raw messages into list[dict(type,value)] for us.
        inputs = [inputs] if isinstance(inputs, str) else inputs
        ark_input = self._build_input(inputs)

        payload: Dict[str, Any] = {
            "model": self.model,
            "input": ark_input,
            "max_output_tokens": kwargs.pop("max_output_tokens", self.max_output_tokens),
        }
        if self.thinking is not None:
            payload["thinking"] = self.thinking

        try:
            resp = self.client.responses.create(**payload)
            # Extract first output_text
            answer = ""
            for out in getattr(resp, "output", []) or []:
                if getattr(out, "type", None) == "message":
                    for c in getattr(out, "content", []) or []:
                        if getattr(c, "type", None) == "output_text" and getattr(c, "text", None):
                            answer = c.text.strip()
                            break
                if answer:
                    break
            if not answer:
                answer = str(resp)
            return 0, answer, "Succeeded!"
        except Exception as err:
            if self.verbose:
                self.logger.error(f"{type(err)}: {err}")
            return -1, self.fail_msg, ""


