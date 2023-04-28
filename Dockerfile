FROM python:3.9.12

# Create non-root group and user
RUN addgroup --system rainfall \
    && adduser --system --home /var/cache/rainfall --ingroup rainfall --uid 1001 dashuser

WORKDIR /usr/share/rainfall/app

COPY requirements.txt /usr/share/rainfall/app/

# Elegantly activating a venv in Dockerfile
ENV VIRTUAL_ENV=/usr/share/rainfall/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install requirements
RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY . /usr/share/rainfall/app/

# set enviroment variables
# This prevent Python from writing out pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# This keeps Python from buffering stdin/stdout
ENV PYTHONUNBUFFERED=1

EXPOSE 80

USER dashuser

ENTRYPOINT ["gunicorn", "app:server", "-b", "0.0.0.0:80", "--workers=4"]