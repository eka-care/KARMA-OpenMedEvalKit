import os
import logging
import tempfile
from io import BytesIO
from typing import List

import librosa
import soundfile as sf
import torch

from karma.models.base_model_abs import BaseModel
from karma.data_models.model_meta import ModelMeta, ModelType, ModalityType
from karma.registries.model_registry import register_model_meta
from karma.data_models.dataloader_iterable import DataLoaderIterable

logger = logging.getLogger(__name__)

def _rnnt_hypotheses_to_texts(hyps, decoding, tokenizer_fallback=None):
    """
    Normalize RNNT decoder outputs to List[str].
    - Accepts: List[str], List[Hypothesis], or List[List[Hypothesis]].
    - Uses RNNT decoder's tokenizer (no script mixing).
    """
    import torch

    if not hyps:
        return []

    # Already strings?
    if isinstance(hyps[0], str):
        return [(s or "").strip() for s in hyps]

    # Pick best per utterance if N-best
    best = [h[0] if isinstance(h, (list, tuple)) else h for h in hyps]

    tok   = getattr(decoding, "tokenizer", None) or tokenizer_fallback
    blank = int(getattr(decoding, "blank_id", 0))

    out = []
    for h in best:
        # Prefer provided text
        t = getattr(h, "text", None)
        if isinstance(t, str) and t.strip():
            out.append(t.strip())
            continue

        # Else decode from token ids
        ids = getattr(h, "y_sequence", None)
        if ids is None:
            ids = getattr(h, "tokens", None)
        if ids is None:
            out.append("")
            continue

        if torch.is_tensor(ids):
            ids = ids.detach().cpu().tolist()
        else:
            ids = list(ids)

        ids = [int(i) for i in ids if i is not None and int(i) >= 0 and int(i) != blank]

        if tok is not None:
            try:
                s = tok.ids_to_text(ids)
                out.append(s.strip() if isinstance(s, str) else str(s))
                continue
            except Exception:
                pass

        # last resort
        out.append(" ".join(map(str, ids)))
    return out

class IndicConformerASR(BaseModel):
    """
    EkaCare Indic Conformer adapter for KARMA.
    Loads RNNT wrapper code + .nemo from the HF repo (no Transformers).
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        language: str = "hindi",
        decoding_method: str = "rnnt",   # we support rnnt greedy path
        target_sample_rate: int = 16000,
        max_symbols_per_step: int = 10,
        **kwargs,
    ):
        super().__init__(model_name_or_path=model_name_or_path, device=device, **kwargs)
        self.language = language
        self.decoding_method = decoding_method
        self.target_sample_rate = target_sample_rate
        self.max_symbols_per_step = max_symbols_per_step

        self.model = None  # will hold RNNTGreedyASR instance
        self.is_loaded = False

    def load_model(self) -> None:
        """Load RNNT wrapper from the HF repo snapshot."""
        import importlib.util
        from huggingface_hub import login, snapshot_download

        # Optional HF login
        tok = os.getenv("HUGGINGFACE_TOKEN")
        if tok:
            try:
                login(tok)
            except Exception as e:
                logger.warning(f"HF login skipped: {e}")

        repo_id = self.model_name_or_path  # e.g. "ekacare/indic-conformer-multilingual-v1"

        # Download wrapper code + weights
        local_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns=["rnnt_asr/**", "converted_model.nemo"],  # change filename if needed
        )

        # Dynamic import: rnnt_asr.RNNTGreedyASR from the snapshot
        mod_path = os.path.join(local_dir, "rnnt_asr", "model.py")
        spec = importlib.util.spec_from_file_location("rnnt_asr.model", mod_path)
        rnnt_mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(rnnt_mod)
        RNNTGreedyASR = rnnt_mod.RNNTGreedyASR

        # Build wrapper from weights on Hub
        self.model = RNNTGreedyASR.from_pretrained(
            repo_id=repo_id,
            filename="converted_model.nemo",               # adjust if different
            max_symbols_per_step=self.max_symbols_per_step,
        )
        # Move underlying NeMo model to device
        self.model.model = self.model.model.to(self.device).eval()
        self.to(self.device)
        self.is_loaded = True

    def run(self, inputs: List[DataLoaderIterable], **kwargs) -> List[str]:
        """Return one transcript per input item."""
        if not self.is_loaded:
            self.load_model()
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        outputs: List[str] = []
        for item in inputs:
            # KARMA passes language in item.other_args (optional)
            lang = (
                (item.other_args or {}).get("language", self.language)[:2].lower()
            )
            _ = kwargs.get("decoding_method", self.decoding_method)  # currently RNNT greedy

            # Preprocess to (1, T) tensor at target sr
            wav = self.preprocess(item)  # (1, T) float32 on self.device

            # RNNTGreedyASR expects file paths; write a small temp WAV
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(
                tmp_path,
                wav.squeeze(0).detach().cpu().numpy(),
                self.target_sample_rate,
            )
            try:
                # Ask for hypotheses (some NeMo versions ignore the flag, this is safest)
                hyps = self.model.transcribe([tmp_path], batch_size=1, return_hypotheses=True)

                # Decode using RNNT decoder's tokenizer (no intra-word mixing)
                dec = getattr(self.model.model, "decoding", None)
                tok = getattr(dec, "tokenizer", None) or getattr(self.model.model, "tokenizer", None)
                texts = _rnnt_hypotheses_to_texts(hyps, decoding=dec, tokenizer_fallback=tok)

                text = texts[0] if texts else ""
                outputs.append((text or "").strip())
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        return outputs

    def preprocess(self, input_item: DataLoaderIterable, **kwargs) -> torch.Tensor:
        """Load bytes → mono float32 tensor (1, T) at target sr, on self.device."""
        audio, _ = librosa.load(BytesIO(input_item.audio), sr=self.target_sample_rate)
        return torch.tensor(audio, dtype=torch.float32, device=self.device).unsqueeze(0)

    def postprocess(self, model_outputs: List[str], **kwargs) -> List[str]:
        """KARMA expects string transcripts; nothing to do."""
        return model_outputs


# -------- Register this model under a distinct name --------
INDIC_CONFORMER_EKA_META = ModelMeta(
    name="ekacare/indic-conformer-multilingual-v1",   # the name you pass to KARMA CLI
    model_type=ModelType.AUDIO_RECOGNITION,
    modalities=[ModalityType.AUDIO],
    description="EkaCare Indic Conformer via NeMo RNNT greedy wrapper",
    loader_class="karma.models.indic_conformer_eka.IndicConformerASR",
    loader_kwargs={
        "language": "hi",
        "device": "cpu",
        "decoding_method": "rnnt",
        "target_sample_rate": 16000,
        "max_symbols_per_step": 10,
    },
    default_eval_kwargs={
        "language": "hi",
        "decoding_method": "rnnt",
    },
    languages=["hin-Deva", "ben-Beng", "tam-Taml", "tel-Telu", "mar-Deva"],
    reference="https://huggingface.co/ekacare/indic-conformer-multilingual-v1",
    release_date="2025-08-20",
    version="1.0",
)

register_model_meta(INDIC_CONFORMER_EKA_META)
