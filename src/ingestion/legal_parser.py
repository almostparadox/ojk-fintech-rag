import re
from typing import List
from src.config import LegalChunk

class LegalParser:
    BAB_PATTERN = re.compile(r'^(BAB\s+[IVXLCDM]+(?:\n(?![P|p]asal)[^\n]+|[^\n]+)?)', re.MULTILINE | re.IGNORECASE)
    PASAL_PATTERN = re.compile(r'^(Pasal\s+\d+)', re.MULTILINE | re.IGNORECASE)

    def clean_text(self, text: str) -> str:
        # Strip running page artifacts and administrative footnotes
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if re.match(r'^(?:-\s*\d+\s*-|\d+\s+dari\s+\d+|Salinan sesuai dengan aslinya)$', stripped, re.IGNORECASE):
                continue
            lines.append(line)
        return "\n".join(lines)

    def parse_text(
        self,
        text: str,
        reg_id: str,
        reg_title: str,
        status: str = "Berlaku"
    ) -> List[LegalChunk]:
        cleaned = self.clean_text(text)
        chunks: List[LegalChunk] = []

        # Find all BAB headings with positions
        bab_matches = list(self.BAB_PATTERN.finditer(cleaned))
        # Find all Pasal headings with positions
        pasal_matches = list(self.PASAL_PATTERN.finditer(cleaned))

        clean_reg_id = re.sub(r'[^a-zA-Z0-9]', '_', reg_id)

        if not pasal_matches:
            # Fallback if no explicit Pasal found: single chunk
            chunks.append(LegalChunk(
                id=f"{clean_reg_id}_ALL",
                reg_id=reg_id,
                reg_title=reg_title,
                status=status,
                bab="",
                pasal="Semua",
                legal_ref=f"{reg_id} Dokumen Lengkap",
                content=cleaned[:4000]
            ))
            return chunks

        def get_bab_for_position(pos: int) -> str:
            current_bab = ""
            for bm in bab_matches:
                if bm.start() <= pos:
                    current_bab = " ".join(bm.group(1).split())
                else:
                    break
            return current_bab

        for i, pm in enumerate(pasal_matches):
            pasal_title = pm.group(1).strip()
            start_idx = pm.start()
            end_idx = pasal_matches[i + 1].start() if i + 1 < len(pasal_matches) else len(cleaned)

            for bm in bab_matches:
                if start_idx < bm.start() < end_idx:
                    end_idx = bm.start()
                    break

            pasal_body = cleaned[start_idx:end_idx].strip()
            bab_title = get_bab_for_position(start_idx)
            
            clean_pasal = re.sub(r'[^a-zA-Z0-9]', '_', pasal_title)
            chunk_id = f"{clean_reg_id}_{clean_pasal}"

            chunks.append(LegalChunk(
                id=chunk_id,
                reg_id=reg_id,
                reg_title=reg_title,
                status=status,
                bab=bab_title,
                pasal=pasal_title,
                legal_ref=f"{reg_id} {pasal_title}",
                content=pasal_body
            ))

        return chunks
