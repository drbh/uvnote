# uvnote Integration Test

This directory contains a simple integration test for uvnote functionality.

## Files

- `report.md` - Test markdown document with Python cells demonstrating all key features
- `test.sh` - Integration test script that validates full workflow
- `README.md` - This file

## Running the Test

```bash
cd test-repo
./test.sh
```

The test validates:

1. **Parsing**: Markdown with Python code blocks and attributes
2. **Execution**: Cell execution with uv run and dependency management  
3. **Caching**: Results are cached and reused appropriately
4. **HTML Generation**: Static HTML output with embedded results
5. **Artifact Management**: Files created by cells are copied to output
6. **CLI Commands**: build, run, clean commands work correctly

## Expected Outputs

After running the test, you should see:

- `site/report.html` - Generated HTML report
- `site/artifacts/` - Directory containing cell outputs
- `.uvnote/cache/` - Cached execution results
- Console output confirming all features work

The test uses deterministic data (seeded random numbers) to ensure reproducible results.