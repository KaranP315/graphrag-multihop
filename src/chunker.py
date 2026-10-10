from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A text chunk derived from a single paragraph or sentence window."""

    id: str
    text: str
    title: str
    doc_id: str
    sentences: list[str] = field(default_factory=list)


class Chunker:
    """Splits paragraphs into chunks for embedding and retrieval."""

    def chunk_paragraphs(self, paragraphs: list[dict]) -> list[Chunk]:
        """Convert each paragraph into a single chunk.

        HotpotQA paragraphs are already short (typically 5-10 sentences),
        so one-to-one mapping works well here.
        """
        chunks: list[Chunk] = []
        for i, para in enumerate(paragraphs):
            doc_id = para["doc_id"]
            sentences = para["sentences"]
            chunk = Chunk(
                id=f"{doc_id}_chunk_{i}",
                text=" ".join(sentences),
                title=para["title"],
                doc_id=doc_id,
                sentences=sentences,
            )
            chunks.append(chunk)
        return chunks

    def chunk_by_sentences(
        self,
        paragraphs: list[dict],
        window: int = 3,
        stride: int = 2,
    ) -> list[Chunk]:
        """Sliding-window chunking over sentences for larger documents.

        Each chunk contains `window` consecutive sentences, advancing
        by `stride` sentences at each step. Useful when paragraphs are
        longer than typical HotpotQA entries.
        """
        chunks: list[Chunk] = []
        chunk_counter = 0

        for para in paragraphs:
            doc_id = para["doc_id"]
            sentences = para["sentences"]

            if len(sentences) <= window:
                # paragraph fits in a single window
                chunks.append(Chunk(
                    id=f"{doc_id}_chunk_{chunk_counter}",
                    text=" ".join(sentences),
                    title=para["title"],
                    doc_id=doc_id,
                    sentences=list(sentences),
                ))
                chunk_counter += 1
                continue

            for start in range(0, len(sentences), stride):
                window_sents = sentences[start : start + window]
                if not window_sents:
                    break
                chunks.append(Chunk(
                    id=f"{doc_id}_chunk_{chunk_counter}",
                    text=" ".join(window_sents),
                    title=para["title"],
                    doc_id=doc_id,
                    sentences=list(window_sents),
                ))
                chunk_counter += 1

                # stop if we've consumed everything
                if start + window >= len(sentences):
                    break

        return chunks
