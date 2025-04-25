FROM python:3.10-slim


# (0) Install SSH client tools (and git, if you're pulling via SSH)
RUN apt-get update && \
    apt-get install -y --no-install-recommends openssh-client git && \
    rm -rf /var/lib/apt/lists/*

# The two following lines are requirements for the Dev Mode to be functional
# Learn more about the Dev Mode at https://huggingface.co/dev-mode-explorers
RUN useradd -m -u 1000 user
WORKDIR /app


# (2) Copy dependencies manifest
COPY --chown=user requirements.txt requirements.txt

# (3) Install dependencies, mounting SSH keys and optional HTTPS creds
RUN --mount=type=secret,id=AGENTEVAL_DEPLOY_KEY,mode=0400,required=true \
    --mount=type=secret,id=ASTABENCH_DEPLOY_KEY,mode=0400,required=true \
    mkdir -p /root/.ssh && chmod 700 /root/.ssh && \
    cat /run/secrets/AGENTEVAL_DEPLOY_KEY > /root/.ssh/id_ed25519 && chmod 600 /root/.ssh/id_ed25519 && \
    cat /run/secrets/ASTABENCH_DEPLOY_KEY > /root/.ssh/id_astabench && chmod 600 /root/.ssh/id_astabench && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts && \
    printf 'Host github.com\n  User git\n  IdentityFile /root/.ssh/id_ed25519\n  IdentityFile /root/.ssh/id_astabench\n  StrictHostKeyChecking no\n' >> /root/.ssh/config && \
    # rewrite all GitHub HTTPS URLs to SSH so nested deps install via SSH
    git config --global url."ssh://git@github.com/".insteadOf "https://github.com/" && \
    pip install --no-cache-dir --upgrade -r requirements.txt

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
