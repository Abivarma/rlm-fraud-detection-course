#!/bin/bash
# Pre-commit validation script for RLM fraud detection course
# This script validates all changes before allowing a commit

set -e

echo "🔍 Running pre-commit validation..."
echo ""

# Check 1: Run all tests
echo "✓ Running test suite..."
source .venv/bin/activate
python -m pytest tests/ -v --tb=short || {
    echo "❌ Tests failed! Fix tests before committing."
    exit 1
}
echo "✅ All tests passed"
echo ""

# Check 2: Validate notebook outputs
echo "✓ Validating notebook outputs..."
for notebook in notebooks/*.ipynb; do
    if [ -f "$notebook" ]; then
        # Check for error outputs in notebook
        if grep -q '"ename":' "$notebook"; then
            echo "❌ Found errors in $notebook"
            echo "   Please re-run notebook without errors"
            exit 1
        fi

        # Check for rate limit errors
        if grep -q "rate_limit_exceeded\|Rate limit reached" "$notebook"; then
            echo "❌ Found rate limit errors in $notebook"
            echo "   Please re-run with proper delays"
            exit 1
        fi
    fi
done
echo "✅ No errors in notebook outputs"
echo ""

# Check 3: Validate metrics files
echo "✓ Validating metrics files..."
if [ -d "results/metrics" ]; then
    for metrics_file in results/metrics/*.json; do
        if [ -f "$metrics_file" ]; then
            # Validate JSON format
            python -c "import json; json.load(open('$metrics_file'))" || {
                echo "❌ Invalid JSON in $metrics_file"
                exit 1
            }
        fi
    done
fi
echo "✅ Metrics files are valid"
echo ""

# Check 4: Verify visualizations exist
echo "✓ Checking visualizations..."
if [ -d "results/visualizations" ]; then
    viz_count=$(find results/visualizations -name "*.png" | wc -l)
    if [ "$viz_count" -eq 0 ]; then
        echo "⚠️  Warning: No visualizations found"
    else
        echo "✅ Found $viz_count visualization(s)"
    fi
fi
echo ""

# Check 5: Validate course plan alignment
echo "✓ Checking alignment with course goals..."
if [ ! -f "docs/COURSE_PLAN.md" ]; then
    echo "⚠️  Warning: COURSE_PLAN.md not found"
else
    echo "✅ Course plan exists"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All validation checks passed!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Safe to commit 🚀"
