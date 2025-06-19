
## Dependencies

### espeak-ng

```
# required by kokoro
apt-get -qq -y install espeak-ng
```

### npm & npx

see [https://nodejs.org/en/download]()


## Run

```
# first make sure you have an audio input&output device available

uv run -m diy_ai_llm
uv run -m diy_ai_agent
uv run -m diy_ai_speaker
