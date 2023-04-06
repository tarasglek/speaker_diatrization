from speaker_diatrization import speaker_diatrization
from fastapi.responses import FileResponse
from fastapi import FastAPI, Query
from typing import Optional

app = FastAPI()

@app.get("/speaker_diatrization")
async def call_speaker_diatrization(
    compressed_file: str = Query(..., description="Path to the compressed file"),
    srt_filename: str = Query(..., description="Path to the SRT filename"),
    num_speakers: int = Query(..., description="Number of speakers"),
    output_file: Optional[str] = Query(None, description="Path to the output file"),
):
    if output_file is None:
        output_file = f"{compressed_file}.{num_speakers}.diat.srt"
    speaker_diatrization(compressed_file, srt_filename, num_speakers, output_file)
    return FileResponse(output_file, media_type="application/octet-stream", filename=output_file)