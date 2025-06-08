import os

# Check if directories exist
print("Checking directories...")
print(f"cadec exists: {os.path.exists('./cadec')}")
print(f"cadec/original exists: {os.path.exists('./cadec/original')}")

# List files in original
if os.path.exists('./cadec/original'):
    files = os.listdir('./cadec/original')
    ann_files = [f for f in files if f.endswith('.ann')]
    print(f"Found {len(ann_files)} .ann files")
    if ann_files:
        print(f"Sample files: {ann_files[:3]}")
        
        # Try to read first file
        first_file = os.path.join('./cadec/original', ann_files[0])
        print(f"\nReading {ann_files[0]}:")
        try:
            with open(first_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:5]  # First 5 lines
                for i, line in enumerate(lines, 1):
                    print(f"Line {i}: {repr(line)}")
        except Exception as e:
            print(f"Error reading file: {e}")
else:
    print("cadec/original directory not found!")