FROM python:3.11-slim-bookworm AS assets
WORKDIR /build
COPY simulation/asset-builder-requirements.txt .
RUN pip install --no-cache-dir -r asset-builder-requirements.txt
COPY scripts/prepare_franka_assets.py .
ARG APPROVED_REFERENCE_ASSETS=false
RUN test "$APPROVED_REFERENCE_ASSETS" = true \
    && python prepare_franka_assets.py --output /reference --approved-reference-assets

FROM nvcr.io/nvidia/isaac-sim:5.1.0
USER root
WORKDIR /app
COPY simulation/requirements.txt /app/simulation/requirements.txt
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -r /app/simulation/requirements.txt
COPY apps/__init__.py apps/__init__.py
COPY apps/api/ apps/api/
COPY contracts/ contracts/
COPY simulation/ simulation/
COPY learning/__init__.py learning/common.py learning/contract.py learning/capture.py learning/
COPY --from=assets /reference/franka.tar /opt/physicalai/reference/franka.tar
COPY --from=assets /reference/bundle.json /opt/physicalai/reference/bundle.json
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
USER 1234:1234
EXPOSE 8443
ENTRYPOINT ["/isaac-sim/python.sh", "-m", "simulation.bootstrap_tls"]
