from pathlib import Path
from typing import Union
import pypdf

def load_document(file_path: Union[str, Path]) -> str:
    """Load and extract text from a PDF or text file.
    
    Args:
        file_path: Path to the target file (.pdf or text-based like .txt, .md).
        
    Returns:
        Extracted text as string.
        
    Raises:
        FileNotFoundError: If file_path does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".pdf":
        reader = pypdf.PdfReader(str(path))
        text_parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
        return "\n".join(text_parts)
    else:
        return path.read_text(encoding="utf-8")
