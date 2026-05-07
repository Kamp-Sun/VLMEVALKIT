from vlmeval.smp import *
from vlmeval.api.base import BaseAPI

headers = 'Content-Type: application/json'


class GeminiWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'gemini-1.0-pro',
                 retry: int = 5,
                 key: str = None,
                 verbose: bool = True,
                 temperature: float = 0.0,
                 system_prompt: str = None,
                 max_tokens: int = 2048,
                 proxy: str = None,
                 backend='genai',
                 project_id='vlmeval',
                 thinking_budget: int = None,  # range from 0 to 24576
                 # see https://ai.google.dev/gemini-api/docs/thinking
                 fps: int = 1,
                 media_resolution: str = None,
                 **kwargs):

        self.model = model
        self.fail_msg = 'Failed to obtain answer via API. '
        # Allow overriding output token budget via env for large prompts/JSON.
        # GenAI backend uses `max_output_tokens`.
        try:
            env_mt = os.environ.get('GEMINI_MAX_TOKENS', '').strip()
            if env_mt:
                max_tokens = int(env_mt)
        except Exception:
            pass
        self.max_tokens = int(max_tokens)
        self.temperature = temperature
        # Optional thinking budget override via env (useful for speed / stability).
        if thinking_budget is None:
            tb = os.environ.get('GEMINI_THINKING_BUDGET', '').strip()
            if tb:
                try:
                    tb_i = int(tb)
                    # Treat 0 as "do not send thinking_config" (some models require thinking mode
                    # and will reject budget=0).
                    thinking_budget = None if tb_i == 0 else tb_i
                except Exception:
                    thinking_budget = None
        self.thinking_budget = thinking_budget
        self.fps = fps
        # for image, high and medium resolution is 258 tokens per image [default], low resolution is 66 tokens per image
        # for video, not support high resolution, medium resolution is 258 tokens per image [default], low resolution is 66 tokens per image  # noqa: E501
        # Optional media resolution override via env: low/medium/high
        self.media_resolution = media_resolution or os.environ.get('GEMINI_MEDIA_RESOLUTION', None)
        if self.media_resolution:
            assert self.media_resolution in ['low', 'medium', 'high']
        if key is None:
            # Prefer GOOGLE_API_KEY; fall back to GEMINI_API_KEY for convenience.
            key = os.environ.get('GOOGLE_API_KEY', None) or os.environ.get('GEMINI_API_KEY', None)
        # Try to load backend from environment variable
        be = os.environ.get('GOOGLE_API_BACKEND', None)
        if be is not None and be in ['genai', 'vertex']:
            backend = be

        assert backend in ['genai', 'vertex']
        if backend == 'genai':
            # We have not evaluated Gemini-1.5 w. GenAI backend
            assert key is not None  # Vertex does not require API Key
            try:
                from google import genai
                from google.genai import types
            except ImportError as e:
                raise ImportError(
                    "Could not import 'google.genai'. Please install it with:\n"
                    "    pip install --upgrade google-genai"
                ) from e
            self.media_resolution_dict = {
                'low': types.MediaResolution.MEDIA_RESOLUTION_LOW,
                'medium': types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                'high': types.MediaResolution.MEDIA_RESOLUTION_HIGH
            }
            self.genai = genai
            self.client = genai.Client(api_key=key)

        self.backend = backend
        self.project_id = project_id
        self.api_key = key

        if proxy is not None:
            proxy_set(proxy)
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    def upload_video_genai(self, video_path):
        from google import genai
        from google.genai import types
        myfile = self.client.files.upload(file=video_path)

        video_part = types.Part.from_uri(
            file_uri=myfile.uri,
            mime_type="video/mp4"
        )

        video_part.video_metadata = types.VideoMetadata(fps=self.fps)

        while True:
            myfile = self.client.files.get(name=myfile.name)
            if myfile.state == "ACTIVE":
                break
            time.sleep(2)

        return video_part

    def build_msgs_genai(self, inputs):
        video_in_msg = False
        video_parts = []
        text_and_images = [] if self.system_prompt is None else [self.system_prompt]

        for inp in inputs:
            if inp['type'] == 'text':
                text_and_images.append(inp['value'])
            elif inp['type'] == 'image':
                text_and_images.append(Image.open(inp['value']))
            elif inp['type'] == 'video':
                video_file = self.upload_video_genai(inp['value'])
                video_parts.append(video_file)
                video_in_msg = True

        messages = video_parts + text_and_images
        return messages, video_in_msg

    def build_msgs_vertex(self, inputs):
        from vertexai.generative_models import Part, Image
        messages = [] if self.system_prompt is None else [self.system_prompt]
        for inp in inputs:
            if inp['type'] == 'text':
                messages.append(inp['value'])
            elif inp['type'] == 'image':
                messages.append(Part.from_image(Image.load_from_file(inp['value'])))
        return messages

    def generate_inner(self, inputs, **kwargs) -> str:
        if self.backend == 'genai':
            from google.genai import types
            assert isinstance(inputs, list)
            model = self.model
            messages, video_in_msg = self.build_msgs_genai(inputs)

            # Configure generation parameters
            config_args = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens
            }
            # set resolution for vision input
            if self.media_resolution:
                if video_in_msg:
                    assert self.media_resolution != 'high', "For video input, only support medium and low resolution"
                config_args["media_resolution"] = self.media_resolution_dict[self.media_resolution]

            # If thinking_budget is specified, add thinking_config
            # By default, Gemini 2.5 Pro will automatically select
            # a thinking budget not exceeding 8192 if not specified.
            if self.thinking_budget is not None:
                config_args["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=self.thinking_budget
                )
            config_args.update(kwargs)

            try:
                resp = self.client.models.generate_content(
                    model=model,
                    contents=messages,
                    config=types.GenerateContentConfig(**config_args)
                )
                # `resp.text` may be None for some responses (e.g., blocked/structured outputs).
                answer = getattr(resp, "text", None)
                if not answer:
                    # Try to extract from candidates/content/parts
                    try:
                        cands = getattr(resp, "candidates", None) or []
                        if cands:
                            content = getattr(cands[0], "content", None)
                            parts = getattr(content, "parts", None) or []
                            texts = []
                            for p in parts:
                                t = getattr(p, "text", None)
                                if t:
                                    texts.append(str(t))
                            answer = "\n".join(texts).strip() if texts else ""
                    except Exception:
                        answer = ""

                if not isinstance(answer, str) or not answer.strip():
                    # Treat empty/None answer as failure so caller can retry/fallback.
                    # Provide a compact debug summary to help diagnose safety blocks / empty candidates.
                    dbg = {"model": self.model}
                    try:
                        pf = getattr(resp, "prompt_feedback", None)
                        if pf is not None:
                            dbg["prompt_feedback"] = str(pf)
                    except Exception:
                        pass
                    try:
                        cands = getattr(resp, "candidates", None) or []
                        dbg["num_candidates"] = len(cands)
                        if cands:
                            dbg["finish_reason"] = str(getattr(cands[0], "finish_reason", ""))
                            dbg["safety_ratings"] = str(getattr(cands[0], "safety_ratings", ""))
                    except Exception:
                        pass
                    return -1, "", json.dumps(dbg, ensure_ascii=False)
                return 0, answer, 'Succeeded! '
            except Exception as err:
                if self.verbose:
                    self.logger.error(f'{type(err)}: {err}')
                    self.logger.error(f'The input messages are {inputs}.')

                return -1, '', ''
        elif self.backend == 'vertex':
            import vertexai
            from vertexai.generative_models import GenerativeModel
            vertexai.init(project=self.project_id, location='us-central1')
            model_name = 'gemini-1.0-pro-vision' if self.model == 'gemini-1.0-pro' else self.model
            model = GenerativeModel(model_name=model_name)
            messages = self.build_msgs_vertex(inputs)
            try:
                resp = model.generate_content(messages)
                answer = resp.text
                return 0, answer, 'Succeeded! '
            except Exception as err:
                if self.verbose:
                    self.logger.error(f'{type(err)}: {err}')
                    self.logger.error(f'The input messages are {inputs}.')

                return -1, '', ''


class Gemini(GeminiWrapper):
    VIDEO_LLM = True

    def generate(self, message, dataset=None):
        return super(Gemini, self).generate(message)
