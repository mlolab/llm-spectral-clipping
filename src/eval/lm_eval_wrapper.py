"""
Wrapper to make GPTBase/Llama models compatible with EleutherAI's lm-evaluation-harness.

Usage:
    from eval.lm_eval_wrapper import CustomLM
    lm = CustomLM(model=model, batch_size=32, device="cuda:0")
    results = lm_eval.evaluator.simple_evaluate(model=lm, tasks=["hellaswag"])
"""

import torch
import torch.nn.functional as F
import tiktoken
from tqdm import tqdm

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval import utils


@register_model("custom")
class CustomLM(TemplateLM):
    """Wrapper for GPTBase/Llama models for lm-evaluation-harness."""

    def __init__(
        self,
        model,
        batch_size=16,
        max_length=None,
        device="cuda:0",
        dtype="bfloat16",
    ):
        super().__init__()
        self._model = model
        self._model.eval()
        self._tokenizer = tiktoken.get_encoding("gpt2")
        self._batch_size = int(batch_size)
        self._max_length = max_length or model.config.sequence_length
        self._device = device
        self._dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]

    @property
    def eot_token_id(self):
        return self._tokenizer.eot_token

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string, add_special_tokens=None, **kwargs):
        return self._tokenizer.encode(string, allowed_special={"<|endoftext|>"})

    def tok_decode(self, tokens, **kwargs):
        if isinstance(tokens, int):
            tokens = [tokens]
        return self._tokenizer.decode(tokens)

    def _model_call(self, inps):
        """Forward pass returning logits for all positions."""
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda" if "cuda" in str(self._device) else "cpu",
            dtype=self._dtype,
        ):
            # Use targets=inps to get logits for ALL positions (not just last).
            # The loss value is irrelevant here.
            out = self._model(inps, targets=inps, get_logits=True)
            return out["logits"]

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kwargs):
        """
        Compute log-likelihood of continuation tokens given context tokens.

        Each request: ((context_str, continuation_str), context_ids, continuation_ids)
        Returns: list of (log_prob, is_greedy) tuples.
        """
        # Sort by descending total length for efficient batching, track original order
        indexed = list(enumerate(requests))
        indexed.sort(key=lambda x: -(len(x[1][1]) + len(x[1][2])))

        res = [None] * len(requests)
        pbar = tqdm(total=len(requests), disable=disable_tqdm, desc="loglikelihood")

        # Process in batches
        for batch_start in range(0, len(indexed), self._batch_size):
            batch = indexed[batch_start : batch_start + self._batch_size]

            inps = []
            cont_toks_list = []
            inplens = []
            orig_indices = []

            for orig_idx, (_, context_enc, continuation_enc) in batch:
                # Concatenate context + continuation
                # Slice: we feed tokens [0..N-1] and predict [1..N]
                # So input is (ctx + cont)[:-1], and we want log-probs for cont tokens
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self._max_length + 1) :][:-1],
                    dtype=torch.long,
                    device=self._device,
                )
                inps.append(inp)
                cont_toks_list.append(continuation_enc)
                inplens.append(inp.shape[0])
                orig_indices.append(orig_idx)

            # Pad to max length in batch (right-padding)
            max_len = max(inp.shape[0] for inp in inps)
            padded = torch.full(
                (len(inps), max_len),
                self.eot_token_id,
                dtype=torch.long,
                device=self._device,
            )
            for i, inp in enumerate(inps):
                padded[i, : inp.shape[0]] = inp

            # Forward pass
            logits = self._model_call(padded)
            log_probs = F.log_softmax(logits, dim=-1)

            for i in range(len(batch)):
                contlen = len(cont_toks_list[i])
                inplen = inplens[i]

                # The continuation tokens start at position len(context_enc) in
                # the original sequence. After the [:-1] slice, the log-probs
                # for predicting continuation token j are at position
                # (inplen - contlen + j) in the logits.
                # i.e., positions [inplen - contlen, inplen)
                cont_log_probs = log_probs[i, inplen - contlen : inplen]
                cont_tokens = torch.tensor(
                    cont_toks_list[i], dtype=torch.long, device=self._device
                )
                # Gather log-probs for actual continuation tokens
                token_log_probs = cont_log_probs.gather(
                    -1, cont_tokens.unsqueeze(-1)
                ).squeeze(-1)
                total_ll = token_log_probs.sum().item()

                # Check greedy: would argmax select the continuation tokens?
                greedy_tokens = cont_log_probs.argmax(dim=-1)
                is_greedy = (greedy_tokens == cont_tokens).all().item()

                res[orig_indices[i]] = (total_ll, bool(is_greedy))
                pbar.update(1)

        pbar.close()
        return res

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        """Compute rolling log-likelihood (for perplexity evaluation)."""
        results = []

        for req in tqdm(requests, disable=disable_tqdm, desc="loglikelihood_rolling"):
            string = req.args[0]
            token_list = self.tok_encode(string)

            # Create rolling windows
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=token_list,
                        prefix_token=self.eot_token_id,
                        max_seq_len=self._max_length,
                        context_len=1,
                    ),
                )
            )

            # Score each window
            windows = [(None,) + w for w in rolling_token_windows]
            window_results = self._loglikelihood_tokens(
                windows, disable_tqdm=True
            )
            total_ll = sum(ll for ll, _ in window_results)
            results.append(total_ll)

        return results

    def generate_until(self, requests, disable_tqdm=False):
        """Generate text until stop sequence."""
        results = []

        for req in tqdm(requests, disable=disable_tqdm, desc="generate_until"):
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", [])
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            context_enc = self.tok_encode(context)
            # Truncate context from left to leave room for generation
            max_ctx_len = self._max_length - max_gen_toks
            if max_ctx_len <= 0:
                max_ctx_len = 1
            context_enc = context_enc[-max_ctx_len:]

            inp = torch.tensor(
                context_enc, dtype=torch.long, device=self._device
            ).unsqueeze(0)

            with torch.no_grad(), torch.amp.autocast(
                device_type="cuda" if "cuda" in str(self._device) else "cpu",
                dtype=self._dtype,
            ):
                out = self._model.generate(
                    inp,
                    max_new_tokens=max_gen_toks,
                    temperature=gen_kwargs.get("temperature", 0.0) or 1e-10,
                    top_k=None,
                )

            # Decode only the generated tokens (strip context)
            gen_tokens = out[0, len(context_enc) :].tolist()
            gen_text = self.tok_decode(gen_tokens)

            # Truncate at first stop sequence
            for stop_seq in until:
                if stop_seq in gen_text:
                    gen_text = gen_text[: gen_text.index(stop_seq)]

            results.append(gen_text)

        return results
