# EIS Studio - production image
#
#   docker build -t eis-studio .                         # ridge DRT only (small)
#   docker build -t eis-studio --build-arg WITH_STAN=1 . # + Bayesian MAP/HMC DRT
#
FROM python:3.12-slim

ARG WITH_STAN=0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501 \
    EIS_STUDIO_STAN_CACHE=/opt/eis_stan

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Optional: CmdStan + pre-compiled bayes_drt2 models (adds ~1 GB, ~10 min build)
COPY eis_studio ./eis_studio
RUN if [ "$WITH_STAN" = "1" ]; then \
        apt-get update && apt-get install -y --no-install-recommends build-essential \
        && rm -rf /var/lib/apt/lists/* \
        && mkdir -p /opt/cmdstan \
        && python -c "import cmdstanpy; cmdstanpy.install_cmdstan(dir='/opt/cmdstan', progress=False)" \
        && export CMDSTAN=$(ls -d /opt/cmdstan/cmdstan-*) \
        && echo "CMDSTAN=$CMDSTAN" > /etc/cmdstan.env \
        && python -c "from eis_studio import drt; drt.precompile()" \
        && chmod -R a+rwX /opt/eis_stan; \
    fi

COPY . .

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8501\")}/_stcore/health')"

# $PORT lets the same image run on Render, Railway, Cloud Run, Azure, Fly.io …
CMD if [ -f /etc/cmdstan.env ]; then export $(cat /etc/cmdstan.env); fi; \
    streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0
