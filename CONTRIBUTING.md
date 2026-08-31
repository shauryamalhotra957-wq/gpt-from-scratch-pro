# Contributing

Install the development extras and run the test suite:

~~~bash
python -m pip install -e ".[dev]"
pytest
~~~

Keep tokenizer, data-split, checkpoint, and model changes independently testable. Record corpus and configuration changes without committing private text, credentials, or untrusted checkpoints.

Educational metrics are not deployment guarantees; document the evaluation split and hardware when reporting results.
