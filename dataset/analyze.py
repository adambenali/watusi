import lizard

with open("data/train.hl") as f:
    lines = f.readlines()
    average_complexity = 0
    average_length = 0
    min_length = float("inf")
    max_length = -float("inf")
    for line in lines:
        line = line.strip()
        tokens = line.strip().split()
        min_length = min(min_length, len(tokens))
        max_length = max(max_length, len(tokens))
        average_length += len(tokens)
        average_complexity += lizard.analyze_file.analyze_source_code("main.c", f"int main(){{{line.strip()}}}" ).function_list[0].cyclomatic_complexity
    
    average_complexity /= len(lines)
    average_length /= len(lines)

    print("Average cyclomatic complexity:", average_complexity)
    print("Average length:", average_length)
    print("Minimum length:", min_length)
    print("Maximum length:", max_length)
    
