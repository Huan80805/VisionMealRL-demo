FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/home/user/app/src \
    MPLCONFIGDIR=/tmp/matplotlib

COPY --chown=user requirements.txt /home/user/app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /home/user/app/requirements.txt

COPY --chown=user . /home/user/app

USER user
EXPOSE 7860

CMD ["python", "-m", "agent.demo_backend", "--host", "0.0.0.0", "--port", "7860", "--run-root", "runs/agent_three_component_2", "--catalog-dir", "artifacts/catalog/three_component/train", "--image-dir", "artifacts/catalog_demo_images"]
