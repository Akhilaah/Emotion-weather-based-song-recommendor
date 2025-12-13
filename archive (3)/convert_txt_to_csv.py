import csv

def convert_txt_to_csv(input_file, output_file):
    with open(input_file, "r") as infile, open(output_file, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["text", "label"])  # header

        for line in infile:
            parts = line.strip().split(";")
            if len(parts) == 2:
                text, label = parts
                writer.writerow([text.strip(), label.strip()])

    print(f"Saved: {output_file}")

# Convert all splits
convert_txt_to_csv("train.txt", "train.csv")
convert_txt_to_csv("test.txt", "test.csv")
convert_txt_to_csv("val.txt", "val.csv")
