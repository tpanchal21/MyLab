# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

FROM python:3.12

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
#ARG UID=10001

# Copy the source code into the container.
COPY AIAssistantWithFlask.py .
COPY requirements.txt .
COPY /docs/Election2024.pdf .
COPY /docs/Resume.pdf .
COPY ./static /app/static
COPY ./templates /app/templates
# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN pip install -r requirements.txt

# Switch to the non-privileged user to run the application.
#USER appuser


# Expose the port that the application listens on.
EXPOSE 5000

# Set environments OPEN_API_KEY


# Run the application.
CMD ["python", "AIAssistantWithFlask.py"]



#DOCKER COMMANDS TO RUN WHILE IN LOCAL

#docker build -t mychatbot .
#docker run -p 5000:5000 mychatbot
