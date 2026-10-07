"""
Load and manage the Qwen2.5-VL-3B GGUF model via llama-cpp-python.

Handles GPU offloading, multimodal projection, and provides a
simple inference interface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)


class QwenVLModel:
    """
    Wrapper around llama-cpp-python for the Qwen2.5-VL-3B GGUF model.

    The model is loaded lazily on first inference to avoid consuming
    GPU memory at import time.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        mmproj_path: Optional[Path] = None,
        n_gpu_layers: Optional[int] = None,
        ctx_size: Optional[int] = None,
    ):
        settings = get_settings()
        self._model_path = model_path or settings.qwen_model_path
        self._mmproj_path = mmproj_path or settings.qwen_mmproj_path
        self._n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else settings.qwen_n_gpu_layers
        self._ctx_size = ctx_size or settings.qwen_ctx_size

        self._llm = None  # lazy-loaded
        self._chat_handler = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Explicitly load the model into memory."""
        if self._llm is not None:
            logger.info("Model already loaded.")
            return

        import os
        import sys

        # Ensure CUDA & llama DLLs are accessible on Windows
        try:
            import torch
        except ImportError:
            pass

        llama_lib = Path(sys.prefix) / "Lib" / "site-packages" / "llama_cpp" / "lib"
        if llama_lib.is_dir() and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(llama_lib))
            except Exception:
                pass

        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler

        logger.info("Loading Qwen2.5-VL model …")
        logger.info("  Model : %s", self._model_path)
        logger.info("  MMProj: %s", self._mmproj_path)
        logger.info("  GPU layers: %d", self._n_gpu_layers)
        logger.info("  Context   : %d", self._ctx_size)

        if not self._model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self._model_path}")
        if not self._mmproj_path.is_file():
            raise FileNotFoundError(f"MMProj file not found: {self._mmproj_path}")

        self._chat_handler = Qwen25VLChatHandler(
            clip_model_path=str(self._mmproj_path),
            verbose=False,
        )

        self._llm = Llama(
            model_path=str(self._model_path),
            chat_handler=self._chat_handler,
            n_ctx=self._ctx_size,
            n_gpu_layers=self._n_gpu_layers,
            verbose=False,
        )

        logger.info("Qwen2.5-VL model loaded successfully.")

    def unload(self) -> None:
        """Release the model from memory."""
        if self._llm is not None:
            del self._llm
            del self._chat_handler
            self._llm = None
            self._chat_handler = None
            logger.info("Model unloaded.")

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a chat completion request with optional images.

        Parameters
        ----------
        system_prompt : system message text.
        user_prompt : user message text.
        image_paths : list of local image file paths to include.
        max_tokens : max tokens to generate.
        temperature : sampling temperature.

        Returns
        -------
        The assistant's response text.
        """
        if not self.is_loaded:
            self.load()

        settings = get_settings()
        max_tokens = max_tokens or settings.qwen_max_tokens
        temperature = temperature if temperature is not None else settings.qwen_temperature

        # Build content list for the user message
        content: list = []

        # Add images first
        if image_paths:
            import base64
            for img_path in image_paths:
                p = Path(img_path)
                if not p.is_file():
                    logger.warning("Image not found, skipping: %s", img_path)
                    continue
                # Resize image to reasonable resolution for VLM (prevents context overflow & speeds up inference)
                from PIL import Image
                import io

                with Image.open(p) as img:
                    max_dim = 448
                    if max(img.size) > max_dim:
                        scale = max_dim / max(img.size)
                        new_size = (int(img.width * scale), int(img.height * scale))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                    # Convert to RGB if needed
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    img_bytes = buf.getvalue()

                b64 = base64.b64encode(img_bytes).decode("utf-8")
                mime = "image/jpeg"

                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })

        # Add text
        content.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        logger.debug(
            "VLM chat — %d images, prompt length=%d chars",
            len(image_paths or []), len(user_prompt),
        )

        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        answer = response["choices"][0]["message"]["content"]
        answer = answer.strip().lstrip(": ")
        logger.debug("VLM response (%d chars): %s", len(answer), answer[:200])
        return answer.strip()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False
