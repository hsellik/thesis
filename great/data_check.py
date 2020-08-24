import json

if __name__ == '__main__':
    data_path = "C:/Users/Kasutaja/projects/Delft/bug-detection/j2graph/dev.txt"
    bugs = 0
    no_bugs = 0
    with open(data_path, encoding="utf8") as f:
        content = f.readlines()
        for line in content:
            json_data = json.loads(line)
            if json_data["has_bug"] == "true":
                bugs += 1
            if json_data["has_bug"] == "false":
                no_bugs += 1
    print(f"bugs: {bugs}")
    print(f"no_bugs: {no_bugs}")
