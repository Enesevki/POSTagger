# src/utils.py

import os
import json
import pickle
from typing import Any, Union

def ensure_dir(path: str) -> None:
    """
    Bir dizin yolunu alır ve mevcut değilse oluşturur.
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_json(obj: Any, path: str, ensure_ascii: bool = False, indent: int = 2) -> None:
    """
    Bir Python objesini JSON olarak kaydeder.
    """
    dirpath = os.path.dirname(path)
    ensure_dir(dirpath)
    with open(path, 'w', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii=ensure_ascii, indent=indent)

def load_json(path: str) -> Any:
    """
    Bir JSON dosyasını yükleyip Python objesi olarak döner.
    """
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)

def save_pickle(obj: Any, path: str) -> None:
    """
    Bir Python objesini pickle formatında kaydeder.
    """
    dirpath = os.path.dirname(path)
    ensure_dir(dirpath)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path: str) -> Any:
    """
    Pickle formatındaki dosyayı yükleyip Python objesi olarak döner.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)
