import subprocess
import sys
import datetime
import numpy as np
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

def convert_time(secs):
    return datetime.timedelta(seconds=round(secs))

def run(cmd):
    print(f">{cmd}", file=sys.stderr)
    subprocess.run(cmd, shell=True, check=True, stderr=subprocess.STDOUT)

def to_wav(compresed_file, output_file):
    # convert to wav
    # force overwrite file
    run(f"ffmpeg -y -i {compresed_file} -ar 16000 -ac 1 -c:a pcm_s16le {output_file}")

def seconds(srt_time):
    return float((srt_time.hours * 60 + srt_time.minutes) * 60 + srt_time.seconds)

def speaker_diatrization(compressed_file, srt_filename, num_speakers):
    embedding_model = PretrainedSpeakerEmbedding( "speechbrain/spkrec-ecapa-voxceleb", )

    audio_file = compressed_file + ".wav"
    to_wav(compressed_file, audio_file)
    import pysrt
    segments = pysrt.open(srt_filename)

    try:
        # Create embedding
        def segment_embedding(segment):
            audio = Audio()
            start = seconds(segment.start)
            end = seconds(segment.end)
            clip = Segment(start, end)
            waveform, _ = audio.crop(audio_file, clip)
            return embedding_model(waveform[None])

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)
        embeddings = np.nan_to_num(embeddings)
        print(f'Embedding shape: {embeddings.shape}')

        # Assign speaker label
        clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i].text = 'SPEAKER ' + str(labels[i] + 1) + ": " + segments[i].text
        segments.save('/tmp/out.srt', encoding='utf-8')
    except Exception as e:
        raise RuntimeError("Error Running inference with local model", e)

if __name__ == "__main__":
    speaker_diatrization(sys.argv[1], sys.argv[2], int(sys.argv[3]))