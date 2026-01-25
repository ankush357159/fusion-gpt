### How to Run Different Files in Google Collab

Because the project is modular, files should be run as modules, not by double-click style execution.

#### General Rule `python -m package.subpackage.filename`

1. Import function and run

```py
from data.raw.data import load_data

dataset = load_data()
```

2. If file.py have `main()` function then run `!python -m data.raw.data`
