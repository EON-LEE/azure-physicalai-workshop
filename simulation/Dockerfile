FROM nvcr.io/nvidia/isaac-sim:5.1.0
USER root
WORKDIR /app
COPY simulation/requirements.txt /app/simulation/requirements.txt
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -r /app/simulation/requirements.txt
COPY apps/__init__.py apps/__init__.py
COPY apps/api/ apps/api/
COPY contracts/ contracts/
COPY simulation/ simulation/
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
EXPOSE 8443
ENTRYPOINT ["/isaac-sim/python.sh", "-m", "simulation.bootstrap_tls"]
