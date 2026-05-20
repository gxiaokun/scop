import json
import pickle
from pathlib import Path
from typing import Any, Union, Iterable, List
import hashlib


class EnhancedJSONEncoder(json.JSONEncoder):

    def default(self, obj):

        if hasattr(obj, "model_dump"):
            return obj.model_dump()

        elif hasattr(obj, "dict"):
            return obj.dict()

        elif hasattr(obj, "__dataclass_fields__"):
            import dataclasses

            return dataclasses.asdict(obj)

        elif hasattr(obj, "__dict__"):
            return obj.__dict__

        return super().default(obj)


class FileHelper:
    @staticmethod
    def ensure_dir(file_path: Union[str, Path]) -> Path:

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def judge_file_exist(file_path: Union[str, Path]) -> bool:

        path = Path(file_path)
        return path.exists() and path.is_file()

    @staticmethod
    def save_text(file_path: Union[str, Path], text: str, encoding="utf-8") -> None:

        path = FileHelper.ensure_dir(file_path)
        path.write_text(text, encoding=encoding)

    @staticmethod
    def load_text(file_path: Union[str, Path], encoding="utf-8") -> str:

        return Path(file_path).read_text(encoding=encoding)

    @staticmethod
    def save_json(
        file_path: Union[str, Path], data: Any, encoding="utf-8", indent=4
    ) -> None:

        path = FileHelper.ensure_dir(file_path)
        with path.open("w", encoding=encoding) as f:

            json.dump(
                data, f, ensure_ascii=False, indent=indent, cls=EnhancedJSONEncoder
            )

    @staticmethod
    def load_json(file_path: Union[str, Path], encoding="utf-8") -> Any:

        with Path(file_path).open("r", encoding=encoding) as f:
            return json.load(f)

    @staticmethod
    def save_jsonl(
        file_path: Union[str, Path], data: Iterable[Any], encoding="utf-8"
    ) -> None:

        path = FileHelper.ensure_dir(file_path)
        with path.open("w", encoding=encoding) as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    @staticmethod
    def save_jsonl_incremental(
        file_path: Union[str, Path], data: Iterable[Any], encoding="utf-8"
    ) -> None:

        path = FileHelper.ensure_dir(file_path)
        with path.open("a", encoding=encoding) as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    @staticmethod
    def load_jsonl(file_path: Union[str, Path], encoding="utf-8") -> List[Any]:

        items = []
        with Path(file_path).open("r", encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    @staticmethod
    def save_pkl(file_path: Union[str, Path], obj: Any) -> None:

        path = FileHelper.ensure_dir(file_path)
        with path.open("wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_pkl(file_path: Union[str, Path]) -> Any:

        with Path(file_path).open("rb") as f:
            return pickle.load(f)
