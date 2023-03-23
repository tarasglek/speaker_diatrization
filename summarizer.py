#!/usr/bin/env python3
import pysrt
import openai
import os
import sys
from langchain.llms import OpenAI

# parse srt file and summarize it via gpt3
def main(srt_file):
    # load srt file
    segments = pysrt.open(srt_file)
    # get all the text
    text = ""
    llm = OpenAI(temperature=0.7,  max_tokens=-1)
    prompt = """You are a very intelligent GPT summarizer. Write a summary of the most important technical details of what was said using concise bullet points. Example summary:`* compilers are used to perform complex optimizations (SSA, autovectoriation)`.  Your turn GPT:"""
    llm = OpenAI(temperature=0.7,  max_tokens=-1)

    i = 0
    last_generated = segments[0].start
    def generate(text_for_completion, stop):
        nonlocal i
        nonlocal last_generated
        nonlocal text
        if (stop == last_generated):
            raise Exception("Too long of a segment for my algorithm to handle")

        text = ""
        print(f"{last_generated}-{stop}: Summaring len:{len(text_for_completion)} ")
        # print(last_generated)
        response = llm(completion).strip()
        print(response)
        last_generated = stop

    while i < len(segments):
        current_segment = segments[i].text
        text += current_segment + "\n"
        completion = text + "\n\n" + prompt
        if (llm.get_num_tokens(completion + current_segment) > 3000):
            generate(text, segments[i].start)
        i += 1
    generate(text, segments[-1].end)

if __name__ == "__main__":
    main(sys.argv[1])