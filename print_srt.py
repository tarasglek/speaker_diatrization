#!/usr/bin/env python3
import sys
import pysrt
import click

def get_speaker(segment):
    return segment.text.split(':')[0]

def get_text(segment):
    return segment.text.split(':')[1].strip()

def combine_speaker_and_text(speaker, text):
    return f"{speaker}: {text}"

@click.command()
@click.argument('srt-file', type=click.Path(exists=True))
@click.option('--compress/--no-compress', default=False, help='Compress output')
@click.option('--output-file', type=click.Path(), default=None, help='Output file')
def main(srt_file, compress, output_file):
    srt = pysrt.open(srt_file)
    if (compress):
        for i in  reversed(range(len(srt))):
            segment = srt[i]
            if i > 0:
                prev_segment = srt[i - 1]
                speaker = get_speaker(segment)
                if prev_segment.end == segment.start and get_speaker(prev_segment) == speaker:
                    prev_segment.text = combine_speaker_and_text(speaker, get_text(prev_segment) + get_text(segment))
                    prev_segment.end = segment.end
                    srt.pop(i)
                    continue
            output = f"{segment.start} {segment.end} {segment.text}"
    if output_file is None:
        for segment in srt:
            prefix =" ".join([str(segment.start), str(segment.end)])
            print(segment.text)
    else:
        srt.save(output_file, encoding='utf-8')

if __name__ == "__main__":
    main()
