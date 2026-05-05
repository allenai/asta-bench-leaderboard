FROM python:3.10-slim

# The two following lines are requirements for the Dev Mode to be functional
# Learn more about the Dev Mode at https://huggingface.co/dev-mode-explorers
RUN useradd -m -u 1000 user
WORKDIR /app


# (2) Copy dependencies manifest
COPY --chown=user requirements.txt requirements.txt

# (3) Install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# (4) Copy in your Gradio app code
COPY . .
RUN mkdir -p /home/user/data && chown -R user:user /home/user/data

# Make the app treat this as non‑debug (so DATA_DIR=/home/user/data)
ENV system=spaces

# (5) Switch to a non-root user
USER user

# (6) Expose Gradio’s default port
EXPOSE 7860

# (7) Launch your app
CMD ["python", "app.py"]
