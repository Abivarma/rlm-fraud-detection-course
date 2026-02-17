# Custom Agent Configuration for RLM Fraud Detection Course

This project uses specialized agents to ensure high-quality deliverables with proper verification.

## Available Agents

### 1. code-implementer
**When to use:** Building new features, implementing agents, writing code
**Description:** Implements code following TDD principles
**Capabilities:**
- Writes comprehensive tests first (TDD)
- Implements code to pass tests
- Follows best practices and design patterns
- Adds proper documentation and type hints

### 2. notebook-runner
**When to use:** Executing Jupyter notebooks with validation
**Description:** Runs notebooks and validates outputs before committing
**Capabilities:**
- Executes notebooks with proper error handling
- Validates all outputs are generated correctly
- Checks for errors, warnings, or rate limits
- Verifies visualizations are created
- Ensures metrics files are saved
**Validation checks:**
- No execution errors in any cell
- All expected output files exist
- Metrics are within reasonable ranges
- Rate limits are respected

### 3. results-validator
**When to use:** Before any git commit
**Description:** Validates all results and artifacts before pushing to GitHub
**Capabilities:**
- Checks notebook execution outputs are correct
- Verifies no error messages in outputs
- Validates metric files have expected structure
- Ensures visualizations are generated
- Confirms data integrity (no NaN where unexpected)
- Reviews commit message quality
**Validation criteria:**
- All tests pass
- No rate limit errors
- Metrics within expected ranges
- Files are properly formatted
- Commit message is descriptive

### 4. goal-alignment-checker
**When to use:** After completing each phase, before moving to next
**Description:** Ensures work aligns with overall project goals
**Capabilities:**
- Reviews completed work against COURSE_PLAN.md
- Checks if deliverables match requirements
- Validates educational value and clarity
- Ensures technical accuracy
- Verifies progression builds on previous phases
**Alignment checks:**
- Matches course plan objectives
- Educational content is clear
- Technical depth is appropriate
- Results demonstrate the intended insights
- Ready for blog post / white paper

### 5. git-committer
**When to use:** After validation passes, to commit changes
**Description:** Creates well-structured commits with proper messages
**Capabilities:**
- Writes clear, conventional commit messages
- Ensures only necessary files are committed
- Verifies .gitignore is respected
- Adds co-author attribution correctly
- Tags commits appropriately

## Workflow for New Features

```
┌─────────────────────┐
│  User Request       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ code-implementer    │ ← Write tests, implement code
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ notebook-runner     │ ← Execute notebooks, generate results
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ results-validator   │ ← Validate all outputs
└──────────┬──────────┘
           │
      ┌────┴────┐
      │  Pass?  │
      └────┬────┘
           │ Yes
           ▼
┌─────────────────────┐
│goal-alignment-check │ ← Verify against course goals
└──────────┬──────────┘
           │
      ┌────┴────┐
      │Aligned? │
      └────┬────┘
           │ Yes
           ▼
┌─────────────────────┐
│  git-committer      │ ← Commit and push
└─────────────────────┘
```

## Validation Rules

### Before Any Commit:
1. ✅ All tests pass
2. ✅ Notebooks execute without errors
3. ✅ No rate limit failures
4. ✅ All expected files generated
5. ✅ Metrics are reasonable
6. ✅ Visualizations created
7. ✅ Work aligns with course goals

### Before Phase Completion:
1. ✅ Agent implementation complete with tests
2. ✅ Notebook runs end-to-end successfully
3. ✅ Streamlit dashboard works
4. ✅ Results match expected patterns
5. ✅ Educational value is clear
6. ✅ Ready to build on in next phase

## Usage Examples

### Implementing a new phase:
```
1. code-implementer: Write tests and implement RAG agent
2. notebook-runner: Execute notebook with validation
3. results-validator: Check all outputs are correct
4. goal-alignment-checker: Verify meets course objectives
5. git-committer: Commit with proper message
```

### Fixing bugs:
```
1. code-implementer: Fix issue and update tests
2. notebook-runner: Re-run affected notebooks
3. results-validator: Ensure fix resolved the issue
4. git-committer: Commit fix
```

## Current Project Context

**Goal:** Build a comprehensive course demonstrating RLM cost reduction for fraud detection
**Target:** 70-94% cost reduction from naive → RAG → RLM
**Deliverables:**
- Jupyter notebooks (executable with results)
- Streamlit dashboards (interactive)
- Test suite (TDD approach)
- White paper (academic rigor)
- Blog posts (accessible explanations)

**Quality Standards:**
- All code tested before implementation
- Notebooks validated before commit
- No errors in GitHub-visible outputs
- Educational and technically accurate
- Results reproducible
