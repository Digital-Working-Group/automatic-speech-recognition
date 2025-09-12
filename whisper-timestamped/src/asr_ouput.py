"""
asr_output.py
Creates the output files based on types requested by the user
"""
import json
import csv

def write_asr_json(prediction, output_dir):
    """
    Writes all output data to a JSON file
    """
    with open(output_dir / "predictions.json", "w") as out_file:
        json.dump(prediction, out_file, indent=4)
    print(f"Outputted to: {output_dir}/predictions.json")

def write_asr_csv(prediction, output_dir):
    rows = [['segment_id', 'text', 'start', 'end', 'confidence']]
    for segment in prediction["segments"]:
        segment_id = segment["id"]
        for word in segment["words"]:
            row = [segment_id, word["text"], word["start"], word["end"], word["confidence"]]
            rows.append(row)
    with open(output_dir / "predictions.csv", "w") as out_file:
        writer = csv.writer(out_file)
        writer.writerows(rows)
    print(f"Outputted to: {output_dir}/predictions.csv")

def write_asr_txt(prediction, output_dir):
    with open(output_dir / "predictions.txt", "w") as out_file:
        out_file.write(prediction["text"])
    print(f"Outputted to: {output_dir}/predictions.txt")