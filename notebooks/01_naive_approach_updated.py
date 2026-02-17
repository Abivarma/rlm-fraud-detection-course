# This script will update the notebook to add delays between API calls
import json

with open('notebooks/01_naive_approach.ipynb', 'r') as f:
    nb = json.load(f)

# Find the scalability test cell and add time.sleep
for cell in nb['cells']:
    if 'source' in cell and isinstance(cell['source'], list):
        source_text = ''.join(cell['source'])
        if 'Test different batch sizes' in source_text and 'for batch_size in batch_sizes' in source_text:
            # Add import time at the top
            cell['source'] = [line for line in cell['source']]
            # Insert time and sleep after try block
            new_source = []
            for i, line in enumerate(cell['source']):
                new_source.append(line)
                if 'except Exception as e:' in line:
                    # Add delay handling before the except
                    indent_spaces = len(line) - len(line.lstrip())
                    new_source.insert(-1, ' ' * (indent_spaces - 4) + '        # Add delay to avoid rate limiting (20s between requests)\n')
                    new_source.insert(-1, ' ' * (indent_spaces - 4) + '        import time\n')
                    new_source.insert(-1, ' ' * (indent_spaces - 4) + '        time.sleep(20)\n')
                    new_source.insert(-1, ' ' * (indent_spaces - 4) + '        \n')
            cell['source'] = new_source

with open('notebooks/01_naive_approach.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("✓ Updated notebook with delays")
