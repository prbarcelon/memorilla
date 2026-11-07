"""Evaluate a trained memory module on macOS (Apple silicon) without vLLM.

The decoder runs in Hugging Face transformers on the MPS device, and choice scoring embeds the generations with the
encoder in transformers as well. Data loading, prompts, the collator and the metrics are the ones ``evaluate.py`` uses.
Before scoring, a few question embeddings computed here are compared with the precomputed ones, which checks that the
transformers encoder matches the vLLM encoder that produced the released embeddings.
"""

import argparse
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset
from huggingface_hub import snapshot_download
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from memorilla.benchmarks import ACCURACY, ACCURACY_HARD, CHOICE, Benchmark, get_benchmarks
from memorilla.data import EVAL_COLUMNS, HARD_COLUMN, QUESTION_COLUMN, MemoryCollator, MemoryDataset, ensure_data
from memorilla.embeddings import EmbeddingStore, ensure_embeddings, load_split_with_embeddings
from memorilla.llm import inject_memory
from memorilla.memory import MemoryModule
from memorilla.metrics import (
    check_scoring_inputs,
    clean_generation,
    evaluation_records,
    score_dataset,
    write_metrics,
    write_predictions,
)
from memorilla.paths import DATA_DIR, DATA_REPO, DEFAULT_DECODER, DEFAULT_ENCODER, MODEL_CACHE_DIR, RESULTS_DIR, resolve
from memorilla.utils import DEFAULT_SEED, prepare_tokenizer, seed_everything

DEFAULT_CHECKPOINT = "memorilla/Memorilla-Qwen3-8B"
ENCODER_BATCH_SIZE = 16
ENCODER_CHECK_ROWS = 8
ENCODER_CHECK_MIN_COSINE = 0.99


def attention_for(device: torch.device) -> str:
    """Pick the attention implementation for a device.

    On MPS, PyTorch 2.7's SDPA returns NaN for left-padded rows of a batch (the decoder then emits "!!!!" and the
    encoder NaN vectors), so MPS uses eager attention.

    Args:
        device: Torch device.

    Returns:
        ``eager`` on MPS, otherwise ``sdpa``.
    """
    return "eager" if device.type == "mps" else "sdpa"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--benchmark", type=str, default="pmv2", help="Registered benchmark name, or 'all'.")
    parser.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Checkpoint directory, or a Hub model repo id."
    )
    parser.add_argument(
        "--subfolder", type=str, default=None, help="Checkpoint folder inside a Hub repo (default: the benchmark name)."
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows of each benchmark.")
    parser.add_argument("--decoder", type=str, default=DEFAULT_DECODER, help="Decoder the checkpoint was trained with.")
    parser.add_argument("--encoder", type=str, default=DEFAULT_ENCODER, help="Encoder of the embeddings.")
    parser.add_argument("--batch_size", type=int, default=16, help="Rows per generation call.")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        choices=["eager", "sdpa"],
        default=None,
        help="Decoder attention (default: eager on MPS, sdpa elsewhere). SDPA on MPS is safe only at --batch_size 1.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device (default: mps when available).")
    parser.add_argument("--output_dir", type=str, default=str(RESULTS_DIR / "mac"), help="Results directory.")
    parser.add_argument("--data_repo", type=str, default=DATA_REPO, help="Hub dataset repo or local directory.")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR), help="Local mirror of a Hub data repo.")
    parser.add_argument("--cache_dir", type=str, default=MODEL_CACHE_DIR, help="Model cache directory.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    return parser.parse_args()


class HFDecoder:
    """A frozen transformers decoder fed prompt embeddings with the memory tokens scattered in.

    Mirrors ``memorilla.vllm.MemoryVLLM``: the embedding matrix is not resized, the placeholder positions are
    replaced with memory tokens, and decoding is greedy.
    """

    def __init__(
        self,
        decoder: str,
        memory: MemoryModule,
        device: torch.device,
        cache_dir: str | None,
        attn_implementation: str | None = None,
    ) -> None:
        """Load the decoder and tokenizer.

        Args:
            decoder: Hugging Face model id or local path of the decoder.
            memory: The memory module.
            device: Device for the decoder and the memory module.
            cache_dir: Model cache directory.
            attn_implementation: Decoder attention; None picks ``attention_for(device)``.
        """
        self.device = device
        self.memory = memory.to(device=device, dtype=torch.bfloat16).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(decoder, cache_dir=cache_dir, trust_remote_code=True)
        self.memory_token_id = prepare_tokenizer(self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(
            decoder,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation or attention_for(device),
        ).to(device)
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        doc_embeds: torch.Tensor,
        doc_padding_mask: torch.Tensor,
        question_embeds: torch.Tensor,
        max_new_tokens: int = 256,
    ) -> list[str]:
        """Generate one completion per row greedily.

        Args:
            input_ids: Left-padded prompt ids ``[batch, seq_len]`` containing the memory placeholders.
            attention_mask: ``[batch, seq_len]`` with 1 for real tokens.
            doc_embeds: Document embeddings ``[batch, num_docs, embedding_dim]``.
            doc_padding_mask: ``[batch, num_docs]`` with True at padding positions.
            question_embeds: Question embeddings ``[batch, embedding_dim]``.
            max_new_tokens: Maximum number of generated tokens.

        Returns:
            The generated texts.
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        memory_tokens = self.memory(
            doc_embeds.to(self.device, dtype=torch.bfloat16),
            doc_padding_mask.to(self.device),
            question_embeds.to(self.device, dtype=torch.bfloat16),
        )
        inputs_embeds = inject_memory(
            self.model.get_input_embeddings()(input_ids), memory_tokens, input_ids, self.memory_token_id
        )
        output = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)


class HFEncoder:
    """A transformers embedding model with the ``embed`` interface of a ``vllm.LLM`` that ``embed_texts`` expects.

    Qwen3-Embedding pools the last token and normalises, as vLLM does for it; no instruction prefix is added.
    """

    def __init__(self, encoder: str, device: torch.device, cache_dir: str | None) -> None:
        """Load the encoder and tokenizer.

        Args:
            encoder: Hugging Face model id or local path of the encoder.
            device: Torch device.
            cache_dir: Model cache directory.
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(encoder, cache_dir=cache_dir, padding_side="left")
        self.model = AutoModel.from_pretrained(
            encoder, cache_dir=cache_dir, torch_dtype=torch.bfloat16, attn_implementation=attention_for(device)
        ).to(device)
        self.model.eval()

    @torch.inference_mode()
    def vectors(self, texts: list[str]) -> np.ndarray:
        """Embed texts.

        Args:
            texts: Texts to embed.

        Returns:
            Float32 array ``[len(texts), D]`` of unit vectors.
        """
        chunks = []
        for start in range(0, len(texts), ENCODER_BATCH_SIZE):
            batch = self.tokenizer(
                texts[start : start + ENCODER_BATCH_SIZE], padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            hidden = self.model(**batch).last_hidden_state[:, -1]
            chunks.append(F.normalize(hidden.float(), dim=-1).cpu().numpy())
        return np.concatenate(chunks)

    def embed(self, texts: list[str], use_tqdm: bool = True) -> list[SimpleNamespace]:
        """Embed texts in the shape of ``vllm.LLM.embed`` outputs.

        Args:
            texts: Texts to embed.
            use_tqdm: Ignored; kept for interface compatibility.

        Returns:
            One object per text whose ``outputs.embedding`` is the vector.
        """
        return [SimpleNamespace(outputs=SimpleNamespace(embedding=vector)) for vector in self.vectors(texts)]


def load_checkpoint(checkpoint: str, subfolder: str, cache_dir: str | None) -> MemoryModule:
    """Load a memory module from a local directory or a Hub repo folder.

    Args:
        checkpoint: Local checkpoint directory or ``memory.pt``, or a Hub model repo id.
        subfolder: Folder inside the Hub repo.
        cache_dir: Model cache directory.

    Returns:
        The memory module.
    """
    path = resolve(checkpoint)
    if not path.exists():
        root = Path(snapshot_download(repo_id=checkpoint, allow_patterns=[f"{subfolder}/*"], cache_dir=cache_dir))
        path = root / subfolder
    return MemoryModule.from_pretrained(path)


def load_target(benchmark: Benchmark, args: argparse.Namespace) -> tuple[Benchmark, Dataset, EmbeddingStore]:
    """Fetch a benchmark's split and embeddings, apply ``--limit`` and check that the split can be scored.

    Args:
        benchmark: The benchmark.
        args: Parsed command-line arguments.

    Returns:
        The benchmark, its (possibly truncated) split and its embedding store.
    """
    root = ensure_data(benchmark.config, [benchmark.split], repo=args.data_repo, data_dir=args.data_dir)
    ensure_embeddings(
        benchmark.config, [benchmark.split], repo=args.data_repo, data_dir=args.data_dir, encoder=args.encoder
    )
    dataset, store = load_split_with_embeddings(benchmark.config, benchmark.split, root, EVAL_COLUMNS, args.encoder)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))  # row i still maps to question embedding i
    if benchmark.scoring == CHOICE:
        has_hard = HARD_COLUMN in dataset.column_names and any(dataset[HARD_COLUMN])
        benchmark = replace(benchmark, metrics=(ACCURACY, ACCURACY_HARD) if has_hard else (ACCURACY,))
    check_scoring_inputs(benchmark, dataset)
    return benchmark, dataset, store


def generate(
    model: HFDecoder, benchmark: Benchmark, dataset: Dataset, store: EmbeddingStore, batch_size: int
) -> list[str]:
    """Generate one answer per row of a split.

    Args:
        model: The memory-conditioned decoder.
        benchmark: Benchmark that sets the prompt and generation budget.
        dataset: The split.
        store: Its embedding store.
        batch_size: Rows per generation call.

    Returns:
        The cleaned generations in row order.
    """
    collator = MemoryCollator(
        model.tokenizer,
        {(benchmark.config, benchmark.split): store},
        num_memories=model.memory.num_memories,
        max_length=benchmark.max_length,
        system_prompt=benchmark.system_prompt,
        suffix=benchmark.suffix,
        train=False,
    )
    items = MemoryDataset([(benchmark.config, benchmark.split, dataset)])
    predictions = []
    for start in tqdm(range(0, len(items), batch_size), desc=benchmark.name):
        batch = collator([items[index] for index in range(start, min(start + batch_size, len(items)))])
        texts = model.generate(**batch, max_new_tokens=benchmark.max_new_tokens)
        predictions.extend(clean_generation(text) for text in texts)
    return predictions


def check_encoder(encoder: HFEncoder, dataset: Dataset, store: EmbeddingStore) -> float:
    """Compare question embeddings computed here with the precomputed ones.

    Args:
        encoder: The transformers encoder.
        dataset: The evaluated split.
        store: Its embedding store.

    Returns:
        The lowest cosine similarity over the checked rows.

    Raises:
        RuntimeError: If the embeddings disagree, so choice scoring would not match the reference setup.
    """
    rows = range(min(ENCODER_CHECK_ROWS, len(dataset)))
    computed = encoder.vectors([str(dataset[row][QUESTION_COLUMN]).strip() for row in rows])
    stored = F.normalize(torch.stack([store.question(row).float() for row in rows]), dim=-1).numpy()
    lowest = float((computed * stored).sum(axis=1).min())
    print(f"encoder check: lowest cosine to the precomputed question embeddings over {len(rows)} rows = {lowest:.4f}")
    if not lowest >= ENCODER_CHECK_MIN_COSINE:  # also catches NaN, which compares false either way
        raise RuntimeError(f"The transformers encoder disagrees with the precomputed embeddings ({lowest:.4f}).")
    return lowest


def main(args: argparse.Namespace) -> None:
    """Generate and score every requested benchmark.

    Args:
        args: Parsed command-line arguments.

    Raises:
        ValueError: If the checkpoint's output width does not match the decoder's hidden size.
    """
    seed_everything(args.seed)
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    targets = [load_target(benchmark, args) for benchmark in get_benchmarks(args.benchmark)]
    output_dir = resolve(args.output_dir)

    jobs = []
    for benchmark, dataset, store in targets:
        memory = load_checkpoint(args.checkpoint, args.subfolder or benchmark.config, args.cache_dir)
        hidden_size = AutoConfig.from_pretrained(args.decoder, cache_dir=args.cache_dir).hidden_size
        if memory.config.output_dim != hidden_size:
            raise ValueError(
                f"The checkpoint produces {memory.config.output_dim}-dimensional memory tokens but {args.decoder} has "
                f"hidden size {hidden_size}; pass the decoder the checkpoint was trained with."
            )
        model = HFDecoder(args.decoder, memory, device, args.cache_dir, args.attn_implementation)
        predictions = generate(model, benchmark, dataset, store, args.batch_size)
        jobs.append((benchmark, dataset, store, predictions, output_dir / benchmark.name))
        write_predictions(output_dir / benchmark.name, evaluation_records(dataset, predictions))
        del model, memory
        if device.type == "mps":
            torch.mps.empty_cache()

    encoder = None
    for benchmark, dataset, store, predictions, directory in jobs:
        engine = None
        if benchmark.scoring == CHOICE:
            encoder = encoder or HFEncoder(args.encoder, device, args.cache_dir)
            check_encoder(encoder, dataset, store)
            engine = encoder
        summary = write_metrics(directory, score_dataset(benchmark, dataset, predictions, engine), len(predictions))
        print(f"{benchmark.name}: {summary}  ->  {directory}")


if __name__ == "__main__":
    main(parse_args())
