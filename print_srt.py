#!/usr/bin/env python3
import sys
import pysrt
import click
from langchain.llms import OpenAI
import json

def get_speaker(segment):
    speaker, _, _ = segment.text.partition(':')
    return speaker.strip() if speaker else ''

def get_text(segment):
    _, _, text = segment.text.partition(':')
    return text.strip() if text else segment.text

def combine_speaker_and_text(speaker, text):
    return f"{speaker}: {text}"

def compress_segments(segments):
    for i in  reversed(range(len(segments))):
        segment = segments[i]
        if i > 0:
            prev_segment = segments[i - 1]
            speaker = get_speaker(segment)
            if prev_segment.end == segment.start and get_speaker(prev_segment) == speaker:
                prev_segment.text = combine_speaker_and_text(speaker, get_text(prev_segment) + get_text(segment))
                prev_segment.end = segment.end
                segments.pop(i)
                continue
    return segments

def remove_third(ls):
    count = max(1, len(ls) // 3)
    return ls[:count] + ls[-count:]

def text_len(ls):
    return sum(map(len, ls))

def id_speakers(disposable_segments, segments):
    speakers = set()
    compressed_segments = compress_segments(disposable_segments)
    for segment in compressed_segments:
        speakers.add(get_speaker(segment))
    lines = list(map(lambda x: x.text, compressed_segments))
    json_prefix = '{"SPEAKER1" : "'
    def gen_prompt(lines):
        text = "\n".join(lines)
        text = text + "figure out speakers, answer in same language as above:\n```json\n" + json_prefix
        return text
    llm = OpenAI(temperature=0.7,  max_tokens=-1)
    while llm.get_num_tokens(gen_prompt(lines)) > 3000 and len(lines) > 2:
        old_len = len(lines)
        lines = remove_third(lines)
        print(f"Had {old_len} lines, now {len(lines)}")
    completion = llm(gen_prompt(lines), stop = "\n```")
    mapping = (json.loads(json_prefix + completion))
    for segment in segments:
        speaker = get_speaker(segment)
        if speaker in mapping:
            segment.text = combine_speaker_and_text(mapping[speaker], get_text(segment))
        else:
            continue
    return segments

@click.command()
@click.argument('srt-file', type=click.Path(exists=True))
@click.option('--compress/--no-compress', default=False, help='Compress output')
@click.option('--output-file', type=click.Path(), default=None, help='Output file')
@click.option('--name-speakers/--no-name-speakers', default=False, help='Name speakers')
def main(srt_file, compress, name_speakers, output_file):
    segments = pysrt.open(srt_file)
    if compress:
        segments = compress_segments(segments)
    if name_speakers:
        segments = id_speakers(pysrt.open(srt_file), segments)
    if output_file is None:
        for segment in segments:
            prefix =" ".join([str(segment.start), str(segment.end)])
            print(segment.text)
    else:
        segments.save(output_file, encoding='utf-8')

if __name__ == "__main__":
    main()
