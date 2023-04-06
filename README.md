conda activate python-3.10
conda install -c conda-forge hmmlearn
virtualenv -p python3.10 .venv --system-site-packages
uvicorn api:app --reload
curl -X GET "http://localhost:8000/speaker_diatrization?compressed_file=meeting.mp4&srt_filename=meeting.srt&num_speakers=2"