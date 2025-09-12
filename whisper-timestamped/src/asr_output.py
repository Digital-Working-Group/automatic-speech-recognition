"""
asr_output.py
Creates the output files based on types requested by the user
"""
import json
import csv

def write_asr(output_types, prediction, output_fname):
    """
    write ASR output based on output_types
    """
    ext_to_func = {'json': write_asr_json, 'csv': write_asr_csv, 'txt': write_asr_txt}
    for ext in output_types:
        output_fp = f'{output_fname}.{ext}'
        ext_to_func[ext](prediction, output_fp)

def write_asr_json(prediction, output_fp):
    """
    Writes all output data to a JSON file
    """
    with open(output_fp, "w") as out_file:
        json.dump(prediction, out_file, indent=4)
    print(f"Wrote: {output_fp}")

def write_asr_csv(prediction, output_fp):
    """
    write ASR as a CSV.
    """
    rows = [['segment_id', 'text', 'start', 'end', 'confidence']]
    for segment in prediction["segments"]:
        segment_id = segment["id"]
        for word in segment["words"]:
            row = [segment_id, word["text"], word["start"], word["end"], word["confidence"]]
            rows.append(row)
    with open(output_fp, "w") as out_file:
        writer = csv.writer(out_file)
        writer.writerows(rows)
    print(f"Wrote: {output_fp}")

def write_asr_txt(prediction, output_fp):
    """
    write ASR as txt
    """
    with open(output_fp, "w") as out_file:
        out_file.write(prediction["text"])
    print(f"Wrote: {output_fp}")
