"""Prompt builder: the same embeddings upstream `Qwen3TTSForConditionalGeneration.generate()` feeds the talker for
voice cloning in ICL mode with non_streaming_mode=False (text fed one token per frame after the prefill), built per
request from cached per-voice parts."""
import torch


class PromptBuilder:
    def __init__(self, m):
        self.m = m; self.core = m.model; self.talker = self.core.talker; self.cfg = self.core.config; self.tc = self.cfg.talker_config
        self.dev = self.talker.device
        with torch.inference_mode():
            ids = torch.tensor([[self.cfg.tts_bos_token_id, self.cfg.tts_eos_token_id, self.cfg.tts_pad_token_id]], device=self.dev)
            self.tts_bos, self.tts_eos, self.tts_pad = self._text(ids).chunk(3, dim=1)          # each [1,1,H]
            self.pad_vec = self.tts_pad[0, 0]

    def _text(self, ids):
        return self.talker.text_projection(self.talker.get_text_embeddings()(ids))

    def _codec(self, ids):
        return self.talker.get_input_embeddings()(torch.tensor(ids, device=self.dev))

    @torch.inference_mode()
    def voice(self, item):
        """item: upstream VoiceClonePromptItem (ICL). Caches everything that depends only on the voice."""
        ref_ids = self.m._tokenize_texts([self.m._build_ref_text(item.ref_text)])[0][:, 3:-2]
        ref_code = item.ref_code.to(self.dev)                                                    # [T,16]
        parts = [self.talker.get_input_embeddings()(ref_code[:, :1])]
        parts += [self.talker.code_predictor.get_input_embeddings()[i - 1](ref_code[:, i:i + 1]) for i in range(1, self.tc.num_code_groups)]
        codec = torch.cat(parts, dim=1).sum(1).unsqueeze(0)                                      # [1,T,H]
        codec = torch.cat([self._codec([[self.tc.codec_bos_id]]), codec], dim=1)                 # [1,1+T,H]
        spk = item.ref_spk_embedding.to(self.dev).to(self.talker.dtype)
        return dict(ref_ids=ref_ids, ref_code=ref_code, codec=codec, spk=spk, ref_text_embed=self._text(ref_ids))

    @torch.inference_mode()
    def build(self, voice, text, language="English"):
        """Returns (prefill [L,H], trailing [T,H] (may be empty), tts_pad [H])."""
        input_id = self.m._tokenize_texts([self.m._build_assistant_text(text)])[0]
        lang = language.lower()
        if lang == "auto":
            pre = [[self.tc.codec_nothink_id, self.tc.codec_think_bos_id, self.tc.codec_think_eos_id]]
        else:
            pre = [[self.tc.codec_think_id, self.tc.codec_think_bos_id, self.tc.codec_language_id[lang], self.tc.codec_think_eos_id]]
        codec_in = torch.cat([self._codec(pre), voice["spk"].view(1, 1, -1), self._codec([[self.tc.codec_pad_id, self.tc.codec_bos_id]])], dim=1)
        role = self._text(input_id[:, :3])
        head = torch.cat((self.tts_pad.expand(-1, codec_in.shape[1] - 2, -1), self.tts_bos), dim=1) + codec_in[:, :-1]
        # one projection call over [ref text, target text], as upstream does (a per-voice cached ref projection differs by bf16 rounding)
        text_embed = torch.cat([self._text(torch.cat([voice["ref_ids"], input_id[:, 3:-5]], dim=-1)), self.tts_eos], dim=1)
        codec = voice["codec"]; tl, cl = text_embed.shape[1], codec.shape[1]
        if tl > cl:
            icl, trailing = text_embed[:, :cl] + codec, text_embed[0, cl:]
        else:
            icl = torch.cat([text_embed] + [self.tts_pad] * (cl - tl), dim=1) + codec
            trailing = text_embed[0, :0]
        return torch.cat((role, head, icl), dim=1)[0], trailing, self.pad_vec
