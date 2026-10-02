===============================================================
  NEXUS — Dataset Format Guide
  Supported formats: .txt, .csv, .json, .jsonl, .xlsx
===============================================================

NEXUS supports multiple dataset formats for training. Place your
dataset files in this folder (datasets/) and run train.bat.

---------------------------------------------------------------
FORMAT 1: Plain Text (.txt)
---------------------------------------------------------------

Use Q: and A: prefixes. Separate each pair with a blank line.

    Q: What is Python?
    A: Python is a high-level programming language.

    Q: What is AI?
    A: AI stands for Artificial Intelligence.

---------------------------------------------------------------
FORMAT 2: CSV (.csv)
---------------------------------------------------------------

Use two columns. You can name them "question" and "answer"
(or "Q" and "A") in the header row, OR just use two columns
without headers (first column = question, second = answer).

    question,answer
    What is Python?,Python is a high-level programming language.
    What is AI?,AI stands for Artificial Intelligence.

Tip: Wrap values in double quotes if they contain commas.

---------------------------------------------------------------
FORMAT 3: JSON (.json)
---------------------------------------------------------------

A JSON array of objects. Each object must have "question" and
"answer" keys (or "Q" and "A").

    [
      {"question": "What is Python?", "answer": "Python is a programming language."},
      {"question": "What is AI?", "answer": "AI stands for Artificial Intelligence."}
    ]

---------------------------------------------------------------
FORMAT 4: JSON Lines (.jsonl)
---------------------------------------------------------------

One JSON object per line. Same key names as JSON format.

    {"question": "What is Python?", "answer": "Python is a programming language."}
    {"question": "What is AI?", "answer": "AI stands for Artificial Intelligence."}

---------------------------------------------------------------
FORMAT 5: Excel (.xlsx)
---------------------------------------------------------------

First column = question, second column = answer.
If the first row contains "question"/"Q" and "answer"/"A",
it will be treated as a header and skipped.

---------------------------------------------------------------
TIPS
---------------------------------------------------------------

- You can mix formats: put .txt, .csv, and .json files in the
  same folder and NEXUS will parse all of them.
- Minimum recommended: 10 Q&A pairs for meaningful training.
- Keep answers concise (1-3 sentences) for best results.
- Run train.bat to start training, or use the three-dot menu > "Train on Dataset"
  in the Quick Menu to train directly from the GUI.

===============================================================
